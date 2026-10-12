#include <math.h>
/* DC motor with position state x=[i,w,th], digital parallel PID (BE integrator, filtered D),
   saturation + clamping anti-windup, measurement chain:
     encoder quantisation (dth = 2pi/counts, speed = backward difference of quantised angle),
     additive noise sequence nz[k], first-order filter (beta), d-sample actuation delay (0..4),
     dom: derivative on measured speed. ITAE on TRUE speed error (trapezoid). */
double sim4(double Kp,double Ki,double Kd,double N,double Ts,double umax,
            const double*Ad,const double*Bd,double r,double TL,int n,int aw,int dom,
            double counts,const double*nz,double beta,int delay,
            double*yout,double*ymout,double*uout){
    double x0=0,x1=0,x2=0,I=0,D=0,ep=0,ymp=0,J=0,fp=0,a=1.0/(1.0+N*Ts);
    double thqp=0,wf=0,ubuf[5]={0,0,0,0,0};
    double dth = counts>0 ? 2*M_PI/counts : 0;
    for(int k=0;k<n;k++){
        double w=x1, t=k*Ts;
        if(!isfinite(w)||fabs(w)>1e4) return 1e6;
        double wm;
        if(counts>0){ double thq=floor(x2/dth)*dth; wm=(k>0)?(thq-thqp)/Ts:0; thqp=thq; }
        else wm=w;
        if(nz) wm+=nz[k];
        if(beta<1){ wf+=beta*(wm-wf); wm=wf; }
        double e=r-wm, In=I+Ki*Ts*e;
        if(dom) D=a*(D-Kd*N*(wm-ymp)); else D=a*(D+Kd*N*(e-ep));
        double v=Kp*e+In+D, u=v;
        if(umax>0){ if(u>umax)u=umax; else if(u<-umax)u=-umax; }
        if(aw && u!=v && e*v>0){ v=Kp*e+I+D; u=v; if(umax>0){ if(u>umax)u=umax; else if(u<-umax)u=-umax; } }
        else I=In;
        double et=r-w, f=t*fabs(et); if(k>0) J+=0.5*(f+fp)*Ts; fp=f;
        for(int j=delay;j>0;j--) ubuf[j]=ubuf[j-1]; ubuf[0]=u; double ua=ubuf[delay];
        if(yout){ yout[k]=w; ymout[k]=wm; uout[k]=ua; }
        double n0=Ad[0]*x0+Ad[1]*x1+Ad[2]*x2+Bd[0]*ua+Bd[1]*TL;
        double n1=Ad[3]*x0+Ad[4]*x1+Ad[5]*x2+Bd[2]*ua+Bd[3]*TL;
        double n2=Ad[6]*x0+Ad[7]*x1+Ad[8]*x2+Bd[4]*ua+Bd[5]*TL;
        x0=n0;x1=n1;x2=n2; ep=e; ymp=wm;
    }
    return J;
}
