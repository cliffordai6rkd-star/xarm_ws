# Nominal ROBOTIS XL/XC-330 visual geometry

Source: ROBOTIS X330 STEP download, linked from the XL330 manual:
https://www.robotis.com/service/download.php?no=1987
https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/

Geometry attribution: ROBOTIS. These are nominal vendor visual meshes, not
measurements of the connected motors or calibrated mass/inertia data.

Converted from `XL,XC-330.stp` using Gmsh 4.15.2, surface mesh size 0.5–1 mm.
`XL330_body.STL` contains STEP volumes 1, 2, 3, 5, 14, 15 (case and connectors).
`XL330_horn.STL` contains volumes 4, 10 (output horn and centre screw).
Optional rear idler and other screws are omitted. Case surfaces are separate
components; this combined visual mesh is not a watertight inertia/collision solid.

Units: millimetres. X is case width, Y points towards the output shaft end.
Rear case seating datum: `[0, -7.5, -19.5]`; front case face: Z=3.5;
output horn mating face: Z=6.5. The case and horn are attached to different
URDF links, respectively the parent and child of each servo joint.
