import fs from 'fs';

class TextCanvas {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.grid = Array(height).fill(null).map(() => Array(width).fill(' '));
    }
    put(x, y, char) {
        if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
            if (this.grid[y][x] === ' ' || char === '*') {
                this.grid[y][x] = char;
            } else if (this.grid[y][x] !== char) {
                this.grid[y][x] = '+';
            }
        }
    }
    drawLine(x0, y0, x1, y1) {
        x0 = Math.round(x0); y0 = Math.round(y0);
        x1 = Math.round(x1); y1 = Math.round(y1);
        let dx = Math.abs(x1 - x0);
        let dy = Math.abs(y1 - y0);
        let sx = (x0 < x1) ? 1 : -1;
        let sy = (y0 < y1) ? 1 : -1;
        let err = dx - dy;

        while (true) {
            let char = '+';
            if (dx === 0) char = '|';
            else if (dy === 0) char = '-';
            else if (sx === sy) char = '\\\\';
            else char = '/';

            this.put(x0, y0, char);
            
            if (x0 === x1 && y0 === y1) break;
            let e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }
    toString() {
        return this.grid.map(row => row.join('')).join('\n');
    }
}

function generateMandala() {
    let size = 70; // Changed to 70 for 140x70 grid
    let canvas = new TextCanvas(size * 2, size);
    let cx = size;
    let cy = size / 2;
    let radius = size / 2 - 2;
    let points = [];
    let numNodes = 24;
    for (let i = 0; i < numNodes; i++) {
        let angle = (i * Math.PI * 2) / numNodes;
        points.push({
            x: cx + Math.cos(angle) * radius * 2,
            y: cy + Math.sin(angle) * radius
        });
    }
    let steps = [1, 5, 7, 11];
    for (let step of steps) {
        for (let i = 0; i < numNodes; i++) {
            let p1 = points[i];
            let p2 = points[(i + step) % numNodes];
            canvas.drawLine(p1.x, p1.y, p2.x, p2.y);
        }
    }
    for (let p of points) canvas.put(Math.round(p.x), Math.round(p.y), '*');
    return canvas.toString();
}

function generateTessellation() {
    let width = 140; // Changed from 160 to 140
    let height = 70; // Changed from 80 to 70
    let canvas = new TextCanvas(width, height);
    let hexRadiusX = 14; // Scaled down
    let hexRadiusY = 7;
    for (let row = 0; row < 6; row++) {
        for (let col = 0; col < 6; col++) {
            let cx = col * hexRadiusX * 1.5 + 10;
            let cy = row * hexRadiusY * 2 + (col % 2 === 1 ? hexRadiusY : 0) + 10;
            for(let i=0; i<6; i++) {
                let a1 = (i * Math.PI) / 3;
                let a2 = ((i+2) * Math.PI) / 3;
                let px1 = cx + Math.cos(a1) * hexRadiusX;
                let py1 = cy + Math.sin(a1) * hexRadiusY;
                let px2 = cx + Math.cos(a2) * hexRadiusX;
                let py2 = cy + Math.sin(a2) * hexRadiusY;
                canvas.drawLine(px1, py1, px2, py2);
                let a3 = ((i+1) * Math.PI) / 3;
                let px3 = cx + Math.cos(a3) * hexRadiusX;
                let py3 = cy + Math.sin(a3) * hexRadiusY;
                canvas.drawLine(px1, py1, px3, py3);
            }
            canvas.put(Math.round(cx), Math.round(cy), '*');
        }
    }
    return canvas.toString();
}

function generateTesseract() {
    let width = 140;
    let height = 70;
    let canvas = new TextCanvas(width, height);
    let drawCube = (cx, cy, s) => {
        let h = s / 2;
        let w = s;
        let pts = [
            {x: cx - w, y: cy - h}, {x: cx + w, y: cy - h},
            {x: cx + w, y: cy + h}, {x: cx - w, y: cy + h},
            {x: cx - w + w/2, y: cy - h - h/2}, {x: cx + w + w/2, y: cy - h - h/2},
            {x: cx + w + w/2, y: cy + h - h/2}, {x: cx - w + w/2, y: cy + h - h/2}
        ];
        canvas.drawLine(pts[0].x, pts[0].y, pts[1].x, pts[1].y);
        canvas.drawLine(pts[1].x, pts[1].y, pts[2].x, pts[2].y);
        canvas.drawLine(pts[2].x, pts[2].y, pts[3].x, pts[3].y);
        canvas.drawLine(pts[3].x, pts[3].y, pts[0].x, pts[0].y);
        canvas.drawLine(pts[4].x, pts[4].y, pts[5].x, pts[5].y);
        canvas.drawLine(pts[5].x, pts[5].y, pts[6].x, pts[6].y);
        canvas.drawLine(pts[6].x, pts[6].y, pts[7].x, pts[7].y);
        canvas.drawLine(pts[7].x, pts[7].y, pts[4].x, pts[4].y);
        canvas.drawLine(pts[0].x, pts[0].y, pts[4].x, pts[4].y);
        canvas.drawLine(pts[1].x, pts[1].y, pts[5].x, pts[5].y);
        canvas.drawLine(pts[2].x, pts[2].y, pts[6].x, pts[6].y);
        canvas.drawLine(pts[3].x, pts[3].y, pts[7].x, pts[7].y);
        for (let p of pts) canvas.put(Math.round(p.x), Math.round(p.y), '*');
        return pts;
    };
    let cx = width / 2;
    let cy = height / 2;
    let outer = drawCube(cx, cy, 30);
    let inner = drawCube(cx + 10, cy + 5, 10);
    for (let i = 0; i < 8; i++) canvas.drawLine(outer[i].x, outer[i].y, inner[i].x, inner[i].y);
    return canvas.toString();
}

const mathFigures = [
    { name: "Complex Mandala", ascii: "\n" + generateMandala() },
    { name: "Hexagonal Tessellation", ascii: "\n" + generateTessellation() },
    { name: "4D Tesseract", ascii: "\n" + generateTesseract() }
];

let jsContent = "export const mathFigures = " + JSON.stringify(mathFigures, null, 2) + ";\n\n";
jsContent += "export const masterCodeString = `\\\\begin{tikzpicture}\\\\draw[domain=0:2*pi,samples=100] plot ({sin(3*\\\\x r)}, {sin(2*\\\\x r)});\\\\draw (0,0) rectangle (1,1);\\\\draw (0.3,0.3) rectangle (1.3,1.3);\\\\draw (0,0)--(0.3,0.3) (1,0)--(1.3,0.3) (1,1)--(1.3,1.3) (0,1)--(0.3,1.3);\\\\draw[domain=0:2*pi,samples=200] plot (\\\\x: {sin(4*\\\\x r)});\\\\draw (0,0) rectangle (3.4,2.1);\\\\draw (2.1,0) arc (0:90:2.1);\\\\draw (2.1,2.1) arc (90:180:1.3);\\\\draw (0,0) ellipse (2 and 1);\\\\draw (0,0) ellipse (0.8 and 0.3);\\\\node[regular polygon, regular polygon sides=5, minimum size=3cm, draw] {};\\\\draw plot[smooth] coordinates {(0,0) (1,2) (2,-1) (-1,-2) (-2,1) (0,0)};\\\\draw (0,0) -- (1,0) -- (1.5,0.86) -- (1,1.73) -- (0,1.73) -- (-0.5,0.86) -- cycle;\\\\draw[domain=-1:1,y domain=-1:1] plot (\\\\x,\\\\y,{\\\\x*\\\\x-\\\\y*\\\\y});\\\\draw[->] (0,0) -- (1,1);\\\\draw[->] (1,0) -- (2,1);\\\\draw (0,0) -- (4,0) -- (2,3.46) -- cycle;\\\\draw (2,0) -- (3,1.73) -- (1,1.73) -- cycle;\\\\draw (0,0) circle (2);\\\\draw (2,0) circle (1);\\\\draw (3,0) circle (0.5);\\\\draw[domain=0:4*pi] plot (\\\\x, {sin(\\\\x r)});\\\\draw[domain=0:4*pi] plot (\\\\x, {sin(\\\\x r + pi)});\\\\draw (0,0) .. controls (1,1) and (2,-1) .. (3,0);\\\\draw (0,0) -- (0,1) -- (-1,2) (0,1) -- (1,2);\\\\end{tikzpicture}`.repeat(150);";

fs.writeFileSync('/Users/antoniomachuca/Library/CloudStorage/GoogleDrive-71117kb@gmail.com/Mi unidad/CUARTO/VERANO/REPO 5/image-to-tikz-engine/frontend/src/data/mathFigures.js', jsContent);
