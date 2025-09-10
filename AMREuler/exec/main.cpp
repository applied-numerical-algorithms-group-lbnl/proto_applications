#include <iostream>
#include <iomanip>
#include "Proto.H"
#include "BoxOp_Euler.H"
#include "InputParser.H"
//#include "TestFunctions.H"
using namespace Proto;

int TIME_STEP = 0;
int PLOT_NUM = 0;

    Array<double,NUMCOMPS>
integrate(AMRData<double,NUMCOMPS>& a_U,double a_dx)
{
    Array<double,NUMCOMPS> retval;
    for (int c = 0; c< NUMCOMPS; c++)
    {
        retval[c] = a_U.integrate(a_dx,c);
    }
    return retval;
};

void checkcons(AMRData<double,NUMCOMPS>& a_U,
        Array<double,NUMCOMPS>& a_consOld,
        double a_dx,
        double normalize,
        string a_str)
{
    auto consNew = integrate(a_U,a_dx);
    cout << a_str << endl;
    for (int c = 0;c < NUMCOMPS;c++)
    {
        auto consdiff = (consNew[c] - a_consOld[c])/normalize;
        cout << c << ":"
            << " cons quantity, consdiff: "
            << consNew[c] << ", "
            << consdiff << endl;
    }
}

template<unsigned int C>
    PROTO_KERNEL_START
void f_eulerInitializeF(
        Point& a_pt,
        Var<double,C>& a_U,
        double a_h,
        double a_gamma,        
        double a_offset)
{
    double r0 = .125;
    double rsq = 0.;
    double amp = .1;
    for (int dir = 0; dir < DIM ; dir++)
    {
        double xcen = fmod(.5 + a_offset,1.); 
        double xdir = a_pt[dir]*a_h + .5*a_h;
        double del; 
        if (xcen > xdir) del = min(abs(xdir - xcen),abs(xdir - xcen + 1.));
        if (xcen <= xdir) del = min(abs(xdir - xcen),abs(xdir - xcen - 1.));
        rsq += pow(del,2);
    }
    double r = sqrt(rsq);
    if (r > r0)
    {
        a_U(0) = 1.;
    }
    else
    {
        a_U(0) = 1.0 + amp*pow(cos(M_PI*r/r0/2),6);
    }
    double ke = 0.;
    for (int dir = 0; dir < DIM; dir++)
    {
        a_U(dir+1) = 0.;
        ke += a_U(dir+1)*a_U(dir+1)/2;
        a_U(dir+1) *= a_U(0);
    }
    double pressure = pow(a_U(0),a_gamma);
    a_U(C-1) = a_gamma*pressure/(a_gamma - 1.) + a_U(0)*ke;
}
PROTO_KERNEL_END(f_eulerInitializeF, f_eulerInitialize);

int main(int argc, char** argv)
{
#ifdef PR_MPI
    MPI_Init(&argc,&argv);
#endif
#ifdef PR_HDF5
    HDF5Handler h5;
#endif
using Proto::pr_out; 
    typedef BoxOp_Euler<double> OP;

    int domainSize = 64;
    int boxSize = 32;
    int refRatio = 2;
    int maxTimesteps = 4096;
    int numLevels = 3;
    int regridBufferSize = 1;
    int regridInterval = 1;
    double maxTime = .25;
    int outputInterval = 16;
    double t0 = 0.;
    //std::array<bool, DIM> periodicity;
    Array<bool, DIM> periodicity = {1,1,1};
    periodicity.fill(true);

    InputArgs args;
    args.add("domainSize",      domainSize);
    args.add("boxSize",         boxSize);
    args.add("refRatio",        refRatio);
    args.add("maxTimesteps",    maxTimesteps);
    args.add("maxTime",         maxTime);
    args.add("outputInterval",  outputInterval);
    args.add("numLevels",       numLevels);
    args.add("regridBufferSize",regridBufferSize);
    args.add("regridInterval"  ,regridInterval);
    args.add("t0",              t0);
    //args.add("maxRefs",         maxRefs);
    args.parse(argc, argv);
    double consError;
    double physDomainSize = 1.0;
    double dxCoarsest = physDomainSize / domainSize;
    
    args.print();

    if (boxSize > domainSize)
      {
        pr_out() << "Input error: boxSize > domainSize. Forcing boxSize == domainSize." << std::endl;
        boxSize = domainSize;
      }
    PR_TIMER_SETFILE(to_string(domainSize) 
                     + "_DIM" + to_string(DIM) + "_NProc" + to_string(numProc())
                     + "AMREuler.time.table");

    ofstream sizeout(to_string(domainSize) 
                     + "_DIM" + to_string(DIM) + "_NProc" + to_string(numProc())
                     + "HWM.curve");
    Point boxSizeVect = Point::Ones(boxSize);
    Box domainBox = Box::Cube(domainSize);
    ProblemDomain domain(domainBox, periodicity);
    DisjointBoxLayout layout(domain, boxSizeVect);
        
    Array<double, DIM> dx;
    dx.fill(physDomainSize / domainSize);
        
    double dt = 0.25 / domainSize;
     
    std::vector<Point> refRatios(numLevels-1, Point::Ones(refRatio));
    AMRGrid grid(layout, refRatios, numLevels);

    double dxLevel = dx[0];
    Point bufferSize = Point::Ones(regridBufferSize);
    double gamma = 1.4;
    if (numLevels > 1)
      {
        for (int lvl = 0; lvl < numLevels-1; lvl++)
          {
            LevelBoxData<double, OP::numState()> initData(grid[lvl], Point::Zeros());
            Operator::initConvolve(initData, f_eulerInitialize, dxLevel, gamma, t0);
            LevelTagData tags(grid[lvl], bufferSize);
            for (auto iter : grid[lvl])
              {
                OP::generateTags(tags[iter], initData[iter]);
              }
            AMRGrid::buffer(tags, bufferSize);
            grid.regrid(tags, lvl);
            dxLevel /= refRatio;
          }
        
        for (int lvl = 2; lvl < numLevels; lvl++)
          {
            grid.enforceNesting2(lvl);
          }
      }

    AMRData<double, OP::numState()> U(grid, OP::ghost());
    Operator::initConvolve(U, dx[0], f_eulerInitialize, gamma, t0);
    U.averageDown();
    AMRRK4<BoxOp_Euler, double> amrrk4(U, dx);
    Array<double,OP::numState()> sum0;
    for (int c = 0;c < OP::numState();c++)
      {
        sum0[c] = U.integrate(dx[0],c);
      }
    auto l1Soln = U.integrateAbs(dx[0]);

    double time = t0;
#ifdef PR_HDF5
    h5.setTime(time);
    h5.setTimestep(dt);
    h5.writeAMRData(dx, U, "AMREuler_U_"+to_string(domainSize)+"_N0");
#endif
    {      
      struct rusage usage;
      getrusage(RUSAGE_SELF, &usage);
      if (Proto::procID() == 0)
        {
          cout << "memory HWM before evolution: " << usage.ru_maxrss << endl;
          int step = 0;
          sizeout << step << " " << usage.ru_maxrss << endl;
        }
      // pr_out() << "memory HWM before evolution: " << usage.ru_maxrss << endl;
    }
    for (int k = 0; ((k < maxTimesteps) && (time < maxTime)); k++)
      {
        TIME_STEP = k;
        PLOT_NUM = 0;
#if 0
        LevelData<double, 1> externalBCData;  // some LevelData you want to access in your BC
        const auto& bc0 = armrk4.op(0).bc();  // get a reference to the boundary condition on level 0
        bc0.setNumLevelData(1);               // tell BC how many levelData variables it should reference
        bc0.setLevelData(0, externalBCData);  // update BC's reference to external data

        bc0.getLevelData(0);
#endif
        amrrk4.advance(dt);
            
        U.averageDown();
        auto sumFinal = integrate(U,dx[0]);             
        if (Proto::procID() == 0)
          {
            string label = "main loop, step "+ to_string(k) ;
            //checkcons(U,sum0,dx[0],l1Soln,label);
                  
          }
        // Memory HWM.
        
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        if (Proto::procID() == 0)
          {
            //sizeout << k << " " << usage.ru_maxrss << endl;
          }      
        time += dt;
#ifdef PR_HDF5
        h5.setTime(time);
#endif
        if ((k+1) % outputInterval == 0)
          {
            int numBoxes = 0;
            for (int ll = 0; ll < U.numLevels();ll++)
              {
                numBoxes += U[ll].numBoxes();
              }
            if ((Proto::procID() == 0)&&(((k+1) % outputInterval) == 0 ))
              {
                std::cout << "n: " << k+1 << " |  time: " << time << 
                  "  |  number of Boxes: " << numBoxes << std::endl;
              }
#ifdef PR_HDF5             
            h5.writeAMRData(dx, U, "AMREuler_U_"+to_string(domainSize)+"_N"+to_string(k+1));
#endif
          }        
      }
    {
           
      struct rusage usage;
      getrusage(RUSAGE_SELF, &usage);
      if (Proto::procID() == 0)
        {
          cout << "memory HWM after evolution: " << usage.ru_maxrss << endl;
        }
      U.averageDown();
    }
  
    PR_TIMER_REPORT();
#ifdef PR_MPI
    MPI_Finalize();
#endif
}
