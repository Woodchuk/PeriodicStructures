import meshio
from dolfin import *
import numpy as np
from numpy.linalg import inv
import cmath as cm
import matplotlib.pyplot as plt
import os, sys, traceback

lf = 25
rr = 15
hw = 1.0
t = 0.1
wg = 22.86
tol = 1.0e-10

eps_wg = 1.0
eta = 377.0

class PEC(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class InputBC(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], -lf, tol)

class WGOutputBC(SubDomain):
    def inside(self, x, on_boundary):
            return on_boundary and near(x[2], lf, tol)


mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 3)
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mvc, "VolumeRegions")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)


info(mesh)
#plot(mesh)
#plt.show()

# Mark boundaries
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(3)
pec = PEC()
pec.mark(sub_domains, 0)
in_port = InputBC()
in_port.mark(sub_domains, 1)
wg_port = WGOutputBC()
wg_port.mark(sub_domains, 2)
File("BoxSubDomains.pvd").write(sub_domains)

df = 0.2
for m in range(16):
#for m in range(1):
#   K = 0.185 + m * dk
   f0 = 8.7
   K = 6.283* (f0 + df * m) / 300.0
   print("k0 = {0:>f}".format(K), flush=True)
   k0 = Constant(K)  
   beta = sqrt(K*K*eps_wg - (pi /wg)*(pi / wg))
   b0 = Constant(beta)
   
# Set up function spaces
# For low order problem
   cell = tetrahedron
   ele_type = FiniteElement('N1curl', cell, 2, variant="integral") # H(curl) element for EM
   V2 = FunctionSpace(mesh, MixedElement([ele_type, ele_type]))
   V = FunctionSpace(mesh, ele_type)
   u_r, u_i = TrialFunctions(V2)
   v_r, v_i = TestFunctions(V2)

#surface integral definitions from boundaries
   ds = Measure('ds', domain = mesh, subdomain_data = sub_domains)
# Volume regions
   dx_wg = Measure('dx', domain = mesh, subdomain_data = cf, subdomain_id = 1)

# with source and sink terms
   u0 = Constant((0.0, 0.0, 0.0)) #PEC definition
   h_src = Expression(('-sin(pi*x[0]/wg)', '0.0', '0.0'), degree = 2, wg = wg)
   e_src = Expression(('0.0', 'sin(pi * x[0] / wg)', '0.0'), degree = 2, wg = wg)
#Boundary condition dictionary
   boundary_conditions = {0: {'PEC' : u0},
                       1: {'InputBC': (h_src, eps_wg)},
                       2: {'WGOutputBC':0}}

   n = FacetNormal(mesh)

#Build PEC boundary conditions for real and imaginary parts
   bcs = []
   for i in boundary_conditions:
       if 'PEC' in boundary_conditions[i]:
          bc = DirichletBC(V2.sub(0), boundary_conditions[i]['PEC'], sub_domains, i)
          bcs.append(bc)
          bc = DirichletBC(V2.sub(1), boundary_conditions[i]['PEC'], sub_domains, i)
          bcs.append(bc)

# Build input BC source term and loading term
   integral_source = []
   integrals_load =[]
   for i in boundary_conditions:
      if 'InputBC' in boundary_conditions[i]:
          r, s = boundary_conditions[i]['InputBC']
          bb1 = 2.0 * k0 * eta * inner(v_i, cross(n, r)) * ds(i) #Factor of two from field equivalence principle
          integral_source.append(bb1)
          bb2 = inner(cross(n, v_i), cross(n, u_r)) * b0 * ds(i)
          integrals_load.append(bb2)
          bb2 = inner(-cross(n, v_r), cross(n, u_i)) * b0 * ds(i)
          integrals_load.append(bb2)

   for i in boundary_conditions:
      if 'WGOutputBC' in boundary_conditions[i]:
          bb2 = inner(cross(n, v_i), cross(n, u_r)) * b0 * ds(i)
          integrals_load.append(bb2)
          bb2 = inner(-cross(n, v_r), cross(n, u_i)) * b0 * ds(i)
          integrals_load.append(bb2)
# for PMC, do nothing. Natural BC.

   af = (inner(curl(v_r), curl(u_r)) + inner(curl(v_i), curl(u_i)) - eps_wg * k0 * k0 * (inner(v_r, u_r) + inner(v_i, u_i))) * dx_wg +\
        sum(integrals_load)
   L = sum(integral_source)

   u1 = Function(V2)
   vdim = u1.vector().size()
   print("Vdim = ", vdim)

   solve(af == L, u1, bcs, solver_parameters = {'linear_solver' : 'mumps'}) 

   u1_r, u1_i = u1.split(True)

   fp = File("EField_r.pvd")
   fp << u1_r
   fp = File("EField_i.pvd")
   fp << u1_i
#   fp = File('WaveFile.pvd')

#   ut = u1_r.copy(deepcopy=True)
#   for i in range(50):
#      ut.vector().zero()
#      ut.vector().axpy(-sin(pi * i / 25.0), u1_i.vector())
#      ut.vector().axpy(cos(pi * i / 25.0), u1_r.vector()) 
#      fp << (ut, i)


   H = interpolate(h_src, V) # Get input field
   P_thru =  assemble((-dot(u1_r,cross(curl(u1_i),n))+dot(u1_i,cross(curl(u1_r),n))) * ds(2))
   P_refl = assemble((-dot(u1_i,cross(curl(u1_r), n)) + dot(u1_r, cross(curl(u1_i), n))) * ds(1))
   P_inc = assemble((dot(H, H) * 0.5 * eta * k0 / b0) * ds(1))
   print("k0 = ", K)
   print("Beta = ", beta)
   print("Incident power at port 1:", P_inc)
   print("Integrated reflected power on port 1:", P_inc - P_refl / (2.0 * K * eta))
   print("Power passing thru WG (port 2):", P_thru / (2.0 * K * eta))
# Generate S parameters
   E = interpolate(e_src, V) #Unnromalized incident E field
   ccr = assemble(-dot(u1_r - E * (eta * k0 / b0), E * (eta * k0 / b0)) * ds(1))
   cci = assemble((dot(u1_i, E) * eta * k0 / b0) * ds(1))
   cc = assemble((dot(E, E) * eta * eta * k0 * k0 / (b0 * b0)) * ds(1))
   S11 = complex(-ccr / cc, -cci / cc)
   print("S11 = {0:<f}+j{1:<f}".format(S11.real, S11.imag))
   ccr = assemble(-dot(u1_r , E * (eta * k0 / b0)) * ds(2))
   cci = assemble((dot(u1_i, E) * eta * k0 / b0) * ds(2))
   S21 = complex(-ccr / cc, -cci / cc)
   print("S21 = {0:<f}+j{1:<f}".format(S21.real, S21.imag))

#   T = np.array([[S21 - S11 * S11 / S21, S11 / S21], [-S11 / S21, 1.0 / S21]])
   Tinv = np.array([[1.0 / S21, -S11 / S21], [S11 / S21, S21 - S11 * S11 / S21]])
   P = np.array([[0.0, 1.0], [1.0, 0.0]])
   P1 = np.matmul(P, Tinv)
   Ti = np.matmul(P1, P)

   gl = np.arccosh((Ti[0,0] + Ti[1,1]) / 2.0)
   print("Phase shift over unit cell = ", gl)

   
sys.exit(0)

