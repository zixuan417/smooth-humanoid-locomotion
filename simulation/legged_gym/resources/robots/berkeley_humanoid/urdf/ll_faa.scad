%scale(1000) import("../meshs/ll_faa.stl");
translate([42, 160, -546])
rotate([0,-10,0])
{
    cube([160,55,12], center=true);
}
translate([100, 160, -503])
rotate([0,-90,0])
{
    cylinder(h=65,r=27);
}
