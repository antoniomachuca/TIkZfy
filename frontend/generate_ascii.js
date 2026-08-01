import fs from 'fs';

function generateShape(type) {
    const width = 200;
    const height = 130;
    const chars = " .,:;!>7?\]}[{)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
    let result = "";
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let nx = (x / width - 0.5) * 4.0;
            let ny = (y / height - 0.5) * 4.0;
            let d = Math.sqrt(nx*nx + ny*ny);
            let z = NaN;
            let draw = false;
            let light = 0;

            if (type === 'torus') {
                if (d > 0.4 && d < 1.6) {
                    z = Math.sqrt(0.36 - Math.pow(d - 1.0, 2));
                    if (!isNaN(z)) { draw = true; light = Math.max(0, nx*0.5 + ny*0.5 + z*0.7); }
                }
            } else if (type === 'sphere') {
                if (d < 1.5) {
                    z = Math.sqrt(2.25 - d*d);
                    if (!isNaN(z)) { draw = true; light = Math.max(0, nx*0.5 + ny*0.5 + z*0.7); }
                }
            } else if (type === 'saddle') {
                z = nx*nx - ny*ny;
                if (Math.abs(z) < 1.5 && d < 1.8) {
                    draw = true; light = Math.max(0, (nx*ny + 0.5));
                }
            } else if (type === 'hyperboloid') {
                z = nx*nx + ny*ny - 0.5;
                if (Math.abs(z) < 1.0 && d < 1.8) {
                    draw = true; light = Math.max(0, (nx - ny + 1.0) / 2.0);
                }
            } else if (type === 'paraboloid') {
                z = d*d - 1.0;
                if (z < 1.0 && d < 1.8) {
                    draw = true; light = Math.max(0, (nx*0.5 - ny*0.5 + 1.0) / 2.0);
                }
            } else if (type === 'ripple') {
                z = Math.sin(d * 6.0) / (d + 0.5);
                if (d < 2.0) {
                    draw = true; light = Math.max(0, (z + 1.0) / 2.0);
                }
            } else if (type === 'sombrero') {
                z = Math.sin(d * 4.0);
                if (d < 2.0) {
                    draw = true; light = Math.max(0, (z*0.5 + nx*0.2 + 0.5));
                }
            } else if (type === 'cone') {
                z = d - 1.0;
                if (d < 1.8) {
                    draw = true; light = Math.max(0, (nx*0.6 + ny*0.3 + 0.8));
                }
            } else if (type === 'cylinder') {
                if (d > 0.8 && d < 1.2) {
                    draw = true; light = Math.max(0, (nx*0.8 + ny*0.2 + 0.5));
                }
            } else if (type === 'egg') {
                let e = (nx*nx + ny*ny) / 1.2;
                if (e < 1.5) {
                    z = Math.sqrt(1.5 - e);
                    if (!isNaN(z)) { draw = true; light = Math.max(0, nx*0.5 + ny*0.5 + z*0.8); }
                }
            } else if (type === 'diamond') {
                if (Math.abs(nx) + Math.abs(ny) < 1.5) {
                    draw = true; light = Math.max(0, nx*0.5 + ny*0.5 + 0.5);
                }
            } else if (type === 'pillow') {
                z = Math.sin(nx * 2) * Math.cos(ny * 2);
                if (d < 2.0) {
                    draw = true; light = Math.max(0, (z + 1.0) / 2.0);
                }
            } else if (type === 'waves') {
                z = Math.sin(nx * 3) + Math.cos(ny * 3);
                if (d < 2.0) {
                    draw = true; light = Math.max(0, (z + 2.0) / 4.0);
                }
            } else if (type === 'peaks') {
                z = Math.exp(-nx*nx - ny*ny) * 2.0;
                if (d < 2.0) {
                    draw = true; light = Math.max(0, z);
                }
            } else if (type === 'mobius') {
                // simple approximation
                let u = Math.atan2(ny, nx);
                let v = d - 1.0;
                if (Math.abs(v) < 0.5) {
                    z = Math.sin(u/2) * v;
                    draw = true; light = Math.max(0, (nx*0.5 + ny*0.5 + z + 1.0)/3.0);
                }
            }

            if (draw) {
                let char_idx = Math.floor(light * chars.length);
                if (char_idx >= chars.length) char_idx = chars.length - 1;
                if (char_idx < 0) char_idx = 0;
                result += chars[char_idx] + chars[char_idx];
            } else {
                result += "  ";
            }
        }
        result += "\n";
    }
    return result;
}

const mathFigures = [
    { name: "Torus", ascii: generateShape('torus') },
    { name: "Sphere", ascii: generateShape('sphere') },
    { name: "Saddle", ascii: generateShape('saddle') },
    { name: "Hyperboloid", ascii: generateShape('hyperboloid') },
    { name: "Paraboloid", ascii: generateShape('paraboloid') },
    { name: "Ripple", ascii: generateShape('ripple') },
    { name: "Sombrero", ascii: generateShape('sombrero') },
    { name: "Cone", ascii: generateShape('cone') },
    { name: "Cylinder", ascii: generateShape('cylinder') },
    { name: "Egg", ascii: generateShape('egg') },
    { name: "Diamond", ascii: generateShape('diamond') },
    { name: "Pillow", ascii: generateShape('pillow') },
    { name: "Waves", ascii: generateShape('waves') },
    { name: "Peaks", ascii: generateShape('peaks') },
    { name: "Mobius", ascii: generateShape('mobius') }
];

let jsContent = "export const mathFigures = " + JSON.stringify(mathFigures, null, 2) + ";\n\n";
jsContent += "export const masterCodeString = `\\\\begin{tikzpicture}\\\\draw[domain=0:2*pi,samples=100] plot ({sin(3*\\\\x r)}, {sin(2*\\\\x r)});\\\\draw (0,0) rectangle (1,1);\\\\draw (0.3,0.3) rectangle (1.3,1.3);\\\\draw (0,0)--(0.3,0.3) (1,0)--(1.3,0.3) (1,1)--(1.3,1.3) (0,1)--(0.3,1.3);\\\\draw[domain=0:2*pi,samples=200] plot (\\\\x: {sin(4*\\\\x r)});\\\\draw (0,0) rectangle (3.4,2.1);\\\\draw (2.1,0) arc (0:90:2.1);\\\\draw (2.1,2.1) arc (90:180:1.3);\\\\draw (0,0) ellipse (2 and 1);\\\\draw (0,0) ellipse (0.8 and 0.3);\\\\node[regular polygon, regular polygon sides=5, minimum size=3cm, draw] {};\\\\draw plot[smooth] coordinates {(0,0) (1,2) (2,-1) (-1,-2) (-2,1) (0,0)};\\\\draw (0,0) -- (1,0) -- (1.5,0.86) -- (1,1.73) -- (0,1.73) -- (-0.5,0.86) -- cycle;\\\\draw[domain=-1:1,y domain=-1:1] plot (\\\\x,\\\\y,{\\\\x*\\\\x-\\\\y*\\\\y});\\\\draw[->] (0,0) -- (1,1);\\\\draw[->] (1,0) -- (2,1);\\\\draw (0,0) -- (4,0) -- (2,3.46) -- cycle;\\\\draw (2,0) -- (3,1.73) -- (1,1.73) -- cycle;\\\\draw (0,0) circle (2);\\\\draw (2,0) circle (1);\\\\draw (3,0) circle (0.5);\\\\draw[domain=0:4*pi] plot (\\\\x, {sin(\\\\x r)});\\\\draw[domain=0:4*pi] plot (\\\\x, {sin(\\\\x r + pi)});\\\\draw (0,0) .. controls (1,1) and (2,-1) .. (3,0);\\\\draw (0,0) -- (0,1) -- (-1,2) (0,1) -- (1,2);\\\\end{tikzpicture}`.repeat(150);\n";

fs.writeFileSync('/Users/antoniomachuca/Library/CloudStorage/GoogleDrive-71117kb@gmail.com/Mi unidad/CUARTO/VERANO/REPO 5/image-to-tikz-engine/frontend/src/data/mathFigures.js', jsContent);
