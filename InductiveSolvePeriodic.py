import meshio
from mpi4py import MPI as nMPI
from dolfin import *
import os, sys, traceback

tol = 1.0e-3


class PEC(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary 

class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 25.0, tol) and on_boundary

    def map(self, x, y):
        if(near(x[2], -25.0, tol)):
                y[0] = x[0]
                y[1] = x[1]
                y[2] = x[2] + 50.0 # The two waveguide ends
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000

class Slave(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], -25.0, tol) and on_boundary

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
pbc = PeriodicBoundary()
pbc.mark(sub_domains, 1)
slv = Slave()
slv.mark(sub_domains, 2)

File("BoxSubDomains.pvd").write(sub_domains)

dk = 0.0125
for m in range(21):
#for m in range(1):
   comm = nMPI.COMM_WORLD
   mpi_rank = comm.Get_rank()

   
   gamma_r = Constant(0.0)
   gamma_i = Constant(dk*m)
# Set up function spaces
# For low order problem
   cell = tetrahedron
   ele_type = FiniteElement('N1curl', cell, 2, variant="integral") # H(curl) element for EM
   V2 = FunctionSpace(mesh, MixedElement([ele_type, ele_type]), constrained_domain=pbc)
   V = FunctionSpace(mesh, ele_type)
   u_r, u_i = TrialFunctions(V2)
   v_r, v_i = TestFunctions(V2)


#surface integral definitions from boundaries
   ds = Measure('ds', domain = mesh, subdomain_data = sub_domains)
# Volume regions
   dx_wg = Measure('dx', domain = mesh, subdomain_data = cf, subdomain_id = 1)

#Boundary condition dictionary
   boundary_conditions = {0: {'PEC' : (0, 0, 0)},
                       1: {'PBC': 1}}

   n = FacetNormal(mesh)
   az = Constant((0, 0, 1))

#Build PEC boundary conditions for real and imaginary parts
   bcs = []
   for i in boundary_conditions:
       if 'PEC' in boundary_conditions[i]:
          bc = DirichletBC(V2.sub(0), boundary_conditions[i]['PEC'], sub_domains, i)
          bcs.append(bc)
          bc = DirichletBC(V2.sub(1), boundary_conditions[i]['PEC'], sub_domains, i)
          bcs.append(bc)

   A = PETScMatrix()
   B = PETScMatrix()

   a = (inner((curl(v_r) + gamma_r * cross(az, v_r) - gamma_i * cross(az, v_i)), \
           (curl(u_r) + gamma_r * cross(az, u_r) - gamma_i * cross(az, u_i))) + \
           inner((curl(v_i) + gamma_r * cross(az, v_i) + gamma_i * cross(az, v_r)), \
           (curl(u_i) + gamma_r * cross(az, u_i) + gamma_i * cross(az, u_r))) \
           ) * dx_wg 
   b = (inner(v_r, u_r) + inner(v_i, u_i)) * dx_wg 
   L_dummy = (inner(Constant((0, 0, 0)), v_r) + inner(Constant((0,0,0)), v_i)) * dx_wg
   assemble_system(a, L_dummy, bcs, A_tensor = A) # Do this to get symmetric application of boundary conditions
   assemble_system(b, L_dummy, bcs, A_tensor = B)
#   for bc in bcs:
#      bc.apply(A)
#      bc.apply(B)

   eigenSolver = SLEPcEigenSolver(A, B)
   eigenSolver.parameters["spectrum"] = "target magnitude"
   eigenSolver.parameters["problem_type"] = "gen_hermitian"
   eigenSolver.parameters["spectral_transform"] = "shift-and-invert"
   eigenSolver.parameters["spectral_shift"] = 0.05
   eigenSolver.parameters["tolerance"] = 1.0e-14
   eigenSolver.parameters["maximum_iterations"] = 250
#   eigenSolver.parameters["solver"] = "arnoldi"
#   eigenSolver.parameters["solver"] = "subspace"
   eigenSolver.parameters["solver"] = "krylov-schur"
   eigenSolver.parameters["verbose"] = True
   print(eigenSolver.parameters.str(True))
   N=5
   eigenSolver.solve(N)

#   VL = VectorFunctionSpace(mesh, "CG", degree = 2, dim = 3)
#   cc = Expression(("cos(gamma * x[2])", "cos(gamma * x[2])", "cos(gamma * x[2])"), degree = 2, gamma = gamma_i)
#   ss = Expression(("sin(gamma * x[2])", "sin(gamma * x[2])", "sin(gamma * x[2])"), degree = 2, gamma = gamma_i)
#   ut = Function(VL)
##   rrP = Function(VL)
##   riP = Function(VL)
##   s1 = interpolate(ss, VL)
##   c1 = interpolate(cc, VL)
##   upr = TrialFunction(VL)
##   vpr = TestFunction(VL)
##   AC = PETScMatrix()
##   bu = PETScVector()
##   bt = PETScVector()
##   a = inner(upr, vpr)*dx_wg
##   assemble(a, tensor = AC)
##   solver = PETScLUSolver("mumps")
##   solver.parameters['symmetric'] = True
#
#   rt = Function(V2)
#   ct = Function(V2)
   for i in range(min(N, eigenSolver.get_number_converged())):
       r, c, rx, cx = eigenSolver.get_eigenpair(i)
#       rt.vector()[:] = rx
#       ct.vector()[:] = cx
#       rr, ri = rt.split(True)
#       ci, cr = rt.split(True)
       if mpi_rank == 0:
          kk = sqrt(r) 
          print("Eigenvalue {0:<f} = {1:<f}".format(i, kk))
#       FileName = f'Evector_r{i}.pvd'
#       rrP = project(rr, VL)
#       riP = project(ri, VL)
##       bbr = inner(vpr, rr)
##       assemble(bbr, tensor = bt)
##       bbi = inner(vpr, ri)
##       assemble(bbi, tensor = bu)
##       solver.solve(A, rrP.vector(), bt)
##       solver.solve(A, riP.vector(), bu)
#       fp = File(FileName)
#       ut.vector()[:] = rrP.vector()[:] * c1.vector()[:] - riP.vector()[:] * s1.vector()[:]
#       fp << ut
#       FileName = f'Evector_uncorr_r{i}.pvd'
#       fp = File(FileName)
#       fp << rrP
#       print("Eigenvalue {0:<f} = {1:<f}".format(i, kk))
#       FileName = f'Evector_i{i}.pvd'
#       fp = File(FileName)
#       ut.vector()[:] = rrP.vector()[:] * s1.vector()[:] + riP.vector()[:] * c1.vector()[:]
#       fp << ut
#       FileName = f'Evector_uncorr_i{i}.pvd'
#       fp = File(FileName)
#       fp << riP
       



sys.exit(0)

