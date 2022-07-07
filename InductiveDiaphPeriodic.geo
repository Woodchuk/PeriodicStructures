// Inductive diaphragm in rect waveguide with periodic end BCs
a = 22.86;  // WR-90
b = 10.16;
w = 10.0;  // diaphragm width
l = 50.0;  // section length
t = 0.10; // Diaphragm thickness

eps = 1.0e-5;
refin = 0.03;

SetFactory("OpenCASCADE");
Box(1) = {0, 0, -l/2, a, b, l};
Box(2) = {0, 0, -t/2, w, b, t};
BooleanDifference(3) = {Volume{1}; Delete;}{Volume{2}; Delete;};

// Periodic boundary
Periodic Surface{7} = {5} Translate{0, 0, 50};

p = Point In BoundingBox{w-eps, -eps, -t/2-eps, w+eps, b+eps, t/2+eps};
MeshSize{p()} = refin;

Physical Volume("Waveguide") = {3};
Mesh.CharacteristicLengthMax = 2.0;
Mesh.CharacteristicLengthMin = 0.02;
Mesh.Algorithm3D = 4;

