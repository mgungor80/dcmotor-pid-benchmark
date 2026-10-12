#include <math.h>
/* DC motor (states i, w), digital parallel PID (backward-Euler I and filtered D),
   voltage saturation with clamping anti-windup, exact ZOH of plant with inputs [V, TL].
   Error e = r - w. Returns ITAE = int t|e| dt (trapezoid). Divergence -> 1e6. */
double sim3(double Kp,double Ki,double Kd,double N,double Ts,double umax,
            const double*Ad,const double*Bd,double r,double TL,int nsteps,int aw,int dom,
            double*yout,double*uout){
    double x0=0,x1=0,I=0,D=0,eprev=0,yprev=0,J=0,fprev=0,a=1.0/(1.0+N*Ts);
    for(int k=0;k<nsteps;k++){
        double y=x1, e=r-y, t=k*Ts;
        if(!isfinite(y)||fabs(y)>1e4) return 1e6;
        double In=I+Ki*Ts*e;
        if(dom) D=a*(D-Kd*N*(y-yprev)); else D=a*(D+Kd*N*(e-eprev));
        double v=Kp*e+In+D, u=v;
        if(umax>0){ if(u>umax)u=umax; else if(u<-umax)u=-umax; }
        if(aw && u!=v && e*v>0){ v=Kp*e+I+D; u=v; if(umax>0){ if(u>umax)u=umax; else if(u<-umax)u=-umax; } }
        else I=In;
        double f=t*fabs(e); if(k>0) J+=0.5*(f+fprev)*Ts; fprev=f;
        if(yout){ yout[k]=y; uout[k]=u; }
        double n0=Ad[0]*x0+Ad[1]*x1+Bd[0]*u+Bd[1]*TL;
        double n1=Ad[2]*x0+Ad[3]*x1+Bd[2]*u+Bd[3]*TL;
        x0=n0; x1=n1; eprev=e; yprev=y;
    }
    return J;
}
