#!/usr/bin/env python3
"""Render the local xArm7/G2 URDF entirely offline using Pinocchio mesh geometry."""
from pathlib import Path
import argparse
import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf',type=Path,default=ROOT/'gello_teleop/models/xarm7_dynamics_g2.urdf')
    parser.add_argument('--output',type=Path,default=ROOT/'gello_teleop/models/xarm7_g2_preview.png')
    args=parser.parse_args()
    path=args.urdf.resolve()
    model=pin.buildModelFromUrdf(str(path)); data=model.createData()
    geom=pin.buildGeomFromUrdf(model,str(path),pin.GeometryType.VISUAL,package_dirs=[str(path.parent)])
    gdata=geom.createData()
    q=np.array([0.,0.,0.,np.pi/2,0.,np.pi/2,-np.pi/2])
    pin.updateGeometryPlacements(model,data,geom,gdata,q)
    fig=plt.figure(figsize=(7,7),facecolor='white')
    ax=fig.add_subplot(111,projection='3d');points=[]
    light=np.array([-.3,-.6,1.]);light/=np.linalg.norm(light)
    for index,obj in enumerate(geom.geometryObjects):
        mesh=obj.geometry
        verts=np.array([mesh.vertex(i) for i in range(mesh.num_vertices)])
        pose=gdata.oMg[index]
        verts=verts@pose.rotation.T+pose.translation
        faces=np.array([[mesh.tri_indices(i)[j] for j in range(3)] for i in range(mesh.num_tris)])
        triangles=verts[faces]
        normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        norm=np.linalg.norm(normal,axis=1);norm[norm==0]=1
        intensity=.5+.4*np.abs((normal/norm[:,None])@light)
        color=np.tile([.78,.79,.8],(len(faces),1))*intensity[:,None]
        ax.add_collection3d(Poly3DCollection(triangles,facecolors=color,edgecolors='none'))
        points.append(verts)
    points=np.concatenate(points);middle=(points.min(0)+points.max(0))/2
    radius=max(np.ptp(points,axis=0))*.6
    ax.set_xlim(middle[0]-radius,middle[0]+radius)
    ax.set_ylim(middle[1]-radius,middle[1]+radius)
    ax.set_zlim(middle[2]-radius,middle[2]+radius)
    ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.view_init(elev=12,azim=-65)
    ax.set_title('xArm7 + G2 (fixed closed, 0.85 rad)')
    fig.savefig(args.output,dpi=150,bbox_inches='tight');plt.close(fig)
    print(args.output)


if __name__=='__main__':
    main()
