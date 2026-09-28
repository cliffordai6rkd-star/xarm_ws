#!/usr/bin/env python3
"""Export a self-contained offline WebGL assembly viewer (no ROS or CAD needed)."""
import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np
import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gello_teleop.build_gello_urdf import MESH_ROOT


def mesh_buffer(path):
    data = Path(path).read_bytes()
    dtype = np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('a', '<u2')])
    faces = np.frombuffer(data, dtype=dtype, offset=84)
    vertices = np.concatenate((faces['v'], np.repeat(faces['n'][:, None, :], 3, axis=1)), axis=2)
    return base64.b64encode(vertices.astype('<f4').tobytes()).decode('ascii')


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>GELLO xArm7 装配草稿</title><style>
body{margin:0;background:#f1f4f8;font:15px system-ui;color:#18222e;display:flex;height:100vh}
aside{width:310px;padding:20px;overflow:auto;background:white}h1{font-size:20px}p{line-height:1.6}
label{display:block;margin-top:15px}input{width:100%}canvas{flex:1;min-width:0}button{padding:8px;margin:8px 4px 0 0}
.note{color:#96500a}output{float:right}</style><aside><h1>GELLO xArm7 装配草稿</h1>
<p class="note">STL 孔位与电机尺寸重建。电机安装朝向、输出盘装配角与实物零位尚待确认；无已测运动范围和惯量。</p>
<p>拖动旋转视角，滚轮缩放。七个滑块只改变屏幕模型；显示范围 ±180° 不代表实物允许行程。此页面不连接硬件。</p>
<button id="zero">虚拟零位</button><button id="bent">返回中位</button><div id="controls"></div>
<p>彩色：打印件；深灰：名义电机外壳；银灰：输出盘。橡皮筋、线缆和扳机未建模。</p>
<p>如果相邻件方向不符，应修正几何文件中的装配角，再确认编码器零位。</p></aside><canvas id="view"></canvas>
<script>const DATA=__DATA__;
const canvas=document.getElementById('view'),gl=canvas.getContext('webgl',{antialias:true});
if(!gl)throw new Error('浏览器未启用 WebGL');
function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,`
attribute vec3 pos;attribute vec3 normal;uniform mat4 model;uniform mat4 vp;varying vec3 n;
void main(){gl_Position=vp*model*vec4(pos,1.);n=mat3(model)*normal;}`));
gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 n;uniform vec3 color;
void main(){float light=.35+.65*abs(dot(normalize(n),normalize(vec3(.4,-.5,1.))));gl_FragColor=vec4(color*light,1.);}`));
gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
const loc={};for(const n of ['model','vp','color'])loc[n]=gl.getUniformLocation(program,n);
const p=gl.getAttribLocation(program,'pos'),n=gl.getAttribLocation(program,'normal');gl.enableVertexAttribArray(p);gl.enableVertexAttribArray(n);gl.enable(gl.DEPTH_TEST);
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
function mul(a,b){let c=Array(16).fill(0);for(let r=0;r<4;r++)for(let k=0;k<4;k++)for(let j=0;j<4;j++)c[4*r+j]+=a[4*r+k]*b[4*k+j];return c}
function pose(x,r){let [a,b,c]=r,sa=Math.sin(a),ca=Math.cos(a),sb=Math.sin(b),cb=Math.cos(b),sc=Math.sin(c),cc=Math.cos(c);return [cc*cb,cc*sb*sa-sc*ca,cc*sb*ca+sc*sa,x[0],sc*cb,sc*sb*sa+cc*ca,sc*sb*ca-cc*sa,x[1],-sb,cb*sa,cb*ca,x[2],0,0,0,1]}
function upload(location,a){gl.uniformMatrix4fv(location,false,new Float32Array([a[0],a[4],a[8],a[12],a[1],a[5],a[9],a[13],a[2],a[6],a[10],a[14],a[3],a[7],a[11],a[15]]))}
function decode(b){const s=atob(b),bytes=new Uint8Array(s.length);for(let i=0;i<s.length;i++)bytes[i]=s.charCodeAt(i);return new Float32Array(bytes.buffer)}
function box(size){const v=[];for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){const corners=[];for(const [u,w]of [[-1,-1],[1,-1],[1,1],[-1,1]]){let a=[0,0,0];a[axis]=sign*size[axis]/2;a[(axis+1)%3]=u*size[(axis+1)%3]/2;a[(axis+2)%3]=w*size[(axis+2)%3]/2;corners.push(a)}for(const i of [0,1,2,0,2,3]){let norm=[0,0,0];norm[axis]=sign;v.push(...corners[i],...norm)}}return new Float32Array(v)}
const colors=[[.60,.65,.70],[.22,.55,.87],[.27,.73,.58],[.92,.65,.26],[.71,.48,.83],[.91,.43,.38],[.22,.73,.78],[.70,.70,.34]];
const objects=[];DATA.geometry.links.forEach((link,i)=>link.visuals.forEach(v=>{let data=v.mesh?decode(DATA.meshes[v.mesh]):box(v.box_m),m=pose(v.xyz,v.rpy);if(v.mesh){const s=DATA.geometry.mesh_scale;for(let r=0;r<3;r++)for(let c=0;c<3;c++)m[r*4+c]*=s}const buf=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buf);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);const color=v.rgba?v.rgba.split(' ').slice(0,3).map(Number):v.mesh?colors[i]:[.23,.25,.29];objects.push({link:i,buf,count:data.length/6,m,color})}));
const reference=DATA.geometry.comparison_reference_q_deg||[90,0,-90,90,0,90,-90];
let q=reference.map(x=>x*Math.PI/180),yaw=-1.0,pitch=.3,distance=.65,target=[0,0,.17];
const jointLabels=['底座转动','肩部俯仰','上臂轴向转动','肘部俯仰','前臂轴向转动','腕部俯仰','手柄轴向转动'];
const sliders=[],outs=[];for(let i=0;i<7;i++){let label=document.createElement('label');label.textContent='J'+(i+1)+' '+jointLabels[i];let out=document.createElement('output');outs.push(out);label.append(out);let s=document.createElement('input');s.type='range';s.min=-180;s.max=180;s.step=1;s.value=0;s.oninput=()=>{q[i]=Number(s.value)*Math.PI/180;draw()};sliders.push(s);label.append(s);document.getElementById('controls').append(label)}
function sub(a,b){return a.map((x,i)=>x-b[i])}function norm(a){const l=Math.hypot(...a);return a.map(x=>x/l)}function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function dot(a,b){return a.reduce((s,x,i)=>s+x*b[i],0)}
function draw(){let w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;canvas.width=w*devicePixelRatio;canvas.height=h*devicePixelRatio;gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(.94,.96,.98,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
const eye=target.map((x,i)=>x+distance*[Math.cos(pitch)*Math.cos(yaw),Math.cos(pitch)*Math.sin(yaw),Math.sin(pitch)][i]);const z=norm(sub(eye,target)),x=norm(cross([0,0,1],z)),y=cross(z,x);const view=[...x,-dot(x,eye),...y,-dot(y,eye),...z,-dot(z,eye),0,0,0,1];const f=1/Math.tan(.5),near=.005,far=5;const proj=[f*h/w,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),2*far*near/(near-far),0,0,-1,0];upload(loc.vp,mul(proj,view));
const worlds=[I()];DATA.geometry.joints.forEach((j,i)=>worlds.push(mul(mul(worlds[i],pose(j.xyz,j.rpy)),pose([0,0,0],[0,0,q[i]]))));for(const o of objects){gl.bindBuffer(gl.ARRAY_BUFFER,o.buf);gl.vertexAttribPointer(p,3,gl.FLOAT,false,24,0);gl.vertexAttribPointer(n,3,gl.FLOAT,false,24,12);upload(loc.model,mul(worlds[o.link],o.m));gl.uniform3fv(loc.color,o.color);gl.drawArrays(gl.TRIANGLES,0,o.count)}sliders.forEach((s,i)=>{s.value=Math.round(q[i]*180/Math.PI);outs[i].textContent=s.value+'°'})}
let drag=null;canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=null;canvas.onpointermove=e=>{if(drag){yaw-=(e.clientX-drag[0])*.01;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];draw()}};canvas.onwheel=e=>{e.preventDefault();distance=Math.max(.08,Math.min(3,distance*Math.exp(e.deltaY*.001)));draw()};
document.getElementById('zero').onclick=()=>{q.fill(0);draw()};document.getElementById('bent').onclick=()=>{q=reference.map(x=>x*Math.PI/180);draw()};new ResizeObserver(draw).observe(canvas);draw();
</script></html>'''


def export_viewer(config, output):
    meshes = {v['mesh']: mesh_buffer(MESH_ROOT / v['mesh'])
              for link in config['links'] for v in link['visuals'] if 'mesh' in v}
    data = json.dumps({'geometry': config, 'meshes': meshes}, separators=(',', ':'))
    with Path(output).open('x') as file:
        file.write(HTML.replace('__DATA__', data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export_viewer(yaml.safe_load(Path(args.geometry).read_text()), args.output)
    print(args.output)
