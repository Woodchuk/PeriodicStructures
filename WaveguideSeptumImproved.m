clear all;

a = 22.86
w = 11.00

kx = pi / a
l = 30.0 # Section length
nmd = 1.0;

eta0 = 377.0

N = 15; # Number of current segments
M = 101; # Number of spectral terms
P = 201; # Number of frequency points
fstart = 8.0; # Starting frequency (GHz)
df = 0.02; # Freq step

# Build spectral terms once
xp = linspace(0, w, N+1);

for m = 1:N+1
   yp = zeros(1,N+1);
   yp(m) = 1.0;
   printf("Basis function = %d\n", m);
   fflush(stdout);
   for n = 1:M
      ker = @(x) interp1(xp, yp, x, "linear").*sin(n*pi*x/a);
      if(m == 1)
         [s(n,m), err, np] = quadcc(ker, 0, w, 1.0e-6, [xp(m+1)]);
      else if(m == N+1)
         [s(n,m), err, np] = quadcc(ker, 0, w, 1.0e-6, [xp(m-1)]);
         else
            [s(n,m), err, np] = quadcc(ker, 0, w, 1.0e-6, [xp(m-1), xp(m), xp(m+1)]);
         endif
      endif
    endfor
 endfor
 for m = 1:M
   Cker = @(x) sin(pi * x / a) .* sin(m * pi * x / a);
   C(m) = 2.0 * quadcc(Cker, w, a, 1.0e-6);
 endfor
 printf("Finished DFT, Compute dispersion curves.\n");
 fflush(stdout);
 # Step thru frequencies
 for q = 0:P
   f0 = fstart + df * q #freq (GHz)
   k0 = 2.0 * pi * f0 / 300.0;
   kk(q+1) = k0;
   for n = 1:M
      g(n) = sqrt(kx * kx * n * n - k0 * k0);
   endfor
   CCoef = sqrt(kx * kx - k0 * k0) / (I * k0 * eta0);
   # Incident field in aperture
   for m = 1:N+1
     for n = 1:M
       st(n, m) = s(n, m) / g(n);
     endfor
   endfor
   
   
 # The matrix elements summing over the spectral terms!
    A = st'*s;
 
 #Source integral (RHS) again summing over the spectral terms
    rhs = (C * CCoef) * st;
 
 #Current on metal septum
    cc = inverse(A) * rhs';
 
 # Calculate S params
   S11 = exp(g(1)*l)*(1.0 + (-(C(1)*CCoef) + s(1,:) * cc)  * I * k0 * eta0 / (a * g(1)))
   S21 = exp(g(1)*l)*(-(C(1)*CCoef) + s(1,:) * cc)  * I * k0 * eta0/ (g(1) * a)
   
# Scattering into second and third modes
   Sm2 = exp(-g(2)*l)*(-(C(2)*CCoef) + s(2,:) * cc)  * I * k0 * eta0/ (-g(2) * a)
   Sm3 = exp(-g(3)*l)*(-(C(3)*CCoef) + s(3,:) * cc)  * I * k0 * eta0/ (-g(3) * a)
   
#T parameters for concatenation
   T(1,1) = S21 - S11*S11/S21;
   T(1,2) = S11 / S21;
   T(2,1) = - T(1,2);
   T(2,2) = 1.0 / S21;
   Pt = [0.0, 1.0; 1.0, 0.0]; # Permutation
   Ti = Pt * inverse(T) * Pt; # Transformed for forward concatenation

   gl(q+1) = acosh((T(1,1) + T(2,2)) / 2);
   fflush(stdout);
 endfor
 fp = fopen("Dispersion.txt", "w");
 prow = 1;
 for q = 1:P+1         
    if(abs(real(gl(q))) < 1.0e-4)
       prow = 1;
       fprintf(fp, "%f %f %f\n", kk(q), real(gl(q)), imag(gl(q)));
     else
       if(prow == 1)
          fprintf(fp, "\n");
          prow = 0;
       endif 
    endif  
 endfor
 fclose(fp);
 