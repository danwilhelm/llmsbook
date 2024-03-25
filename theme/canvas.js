function drawLine(ctx, x0,y0, x1,y1) {
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
}

function hLine(ctx, y, xpad, width) {
    drawLine(ctx, xpad,y, width-xpad,y);
}

function vLine(ctx, x, ypad, height) {
    drawLine(ctx, x,ypad, x,height-ypad);
}

function drawArrow(ctx, x0,y0, x1,y1, text="", headlen=5) {
    // This makes it so the end of the arrow head is located at tox, toy, don't ask where 1.15 comes from
    var angle = Math.atan2(y1 - y0, x1 - x0);
    var lineWidth = ctx.lineWidth;
    x1 -= Math.cos(angle) * lineWidth * 1.15;
    y1 -= Math.sin(angle) * lineWidth * 1.15;

    //starting path of the arrow from the start square to the end square and drawing the stroke
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);

    // arrow head
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1-headlen*Math.cos(angle-Math.PI/7),y1-headlen*Math.sin(angle-Math.PI/7));
    ctx.lineTo(x1-headlen*Math.cos(angle+Math.PI/7),y1-headlen*Math.sin(angle+Math.PI/7));
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1-headlen*Math.cos(angle-Math.PI/7),y1-headlen*Math.sin(angle-Math.PI/7));

    if (text != "") {
        ctx.fillText(text, x1-lineWidth*2,y1-lineWidth*4)
    }

    //draws the paths created above
    ctx.stroke();
    ctx.fill();
}


function drawCanvas(ctx, width, height) {
    const classOffset = width / 20;
    console.log(width, height);

    ctx.save();
    const scale = 0.8;
    ctx.scale(scale, scale);
    ctx.translate(width*(1-scale)/2, height*(1-scale)/2);

    ctx.fillStyle = "rgb(50 255 50)";
    ctx.strokeStyle = "rgb(50 255 50)";
    ctx.lineWidth = 1;
    ctx.lineCap = "round"
    ctx.lineDashOffset = 0;
    ctx.setLineDash([4, 2]);
    ctx.globalAlpha = 0.5;

    const nGridLines = 5;
    const xspacing = width / nGridLines;
    const yspacing = height / nGridLines;
    ctx.beginPath();
    for (let i = 1; i < nGridLines; i++) {
        vLine(ctx, i*xspacing, 0, height);
        hLine(ctx, i*yspacing, 0, width);
    }
    ctx.stroke();

    // Class vectors
    ctx.font = "20px sans-serif";
    ctx.fillStyle = "rgb(50 255 50)";
    ctx.strokeStyle = "rgb(50 255 50)";
    ctx.lineWidth = 5;
    ctx.globalAlpha = 1.0;
    ctx.setLineDash([]);
    drawArrow(ctx, 0,height, 3*width/4,height-classOffset, "rot 0 class");
    drawArrow(ctx, 0,height, width/16,classOffset, "rot 5 class");
    ctx.fillStyle = "rgb(255 50 50)";
    ctx.strokeStyle = "rgb(255 50 50)";
    drawArrow(ctx, 0,height, width/8,classOffset*2, "ciphertext");

    // Axes
    ctx.fillStyle = "rgb(255 255 255)";
    ctx.strokeStyle = "rgb(255 255 255)";
    ctx.lineWidth = 3;
    drawArrow(ctx, 0,height, width,height, "'e' freq");
    drawArrow(ctx, 0,height, 0,0, "'b' freq");

    ctx.restore();
}


function resizeCanvas() {
    const canvas = document.getElementById("unembedding-vectors");
    if (canvas && canvas.getContext) {
        const ctx = canvas.getContext("2d");
        const scale = window.devicePixelRatio;
        const width = scale * canvas.clientWidth;   // clientWidth
        const height = scale * canvas.clientHeight;
        canvas.width = width;
        canvas.height = height;
        
        
        const xpad = 0; // (width*(1-scale)) / 2;
        const ypad = 0; // (height*(1-scale)) / 2;
        console.log(scale, width, height);

        ctx.clearRect(0,0, canvas.width,canvas.height);
        ctx.save();
            ctx.scale(scale, scale);
            ctx.translate(xpad, ypad);
            drawCanvas(ctx, width / scale - 2*xpad, height / scale - 2*ypad);
        ctx.restore();
    }
}

window.addEventListener("load", resizeCanvas);
window.addEventListener('resize', resizeCanvas);
