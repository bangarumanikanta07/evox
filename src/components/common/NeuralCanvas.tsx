import React, { useEffect, useRef } from 'react';

interface NeuralCanvasProps {
  className?: string;
  activeStage?: number; // 0: DATA, 1: ANALYZE, 2: OPTIMIZE, 3: EXPLAIN, 4: DECIDE
}

export const NeuralCanvas: React.FC<NeuralCanvasProps> = ({ className = '', activeStage = 0 }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || 800);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 400);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener('resize', handleResize);

    // Particle nodes
    const nodeCount = Math.min(50, Math.floor(width / 22));
    const nodes: {
      x: number;
      y: number;
      vx: number;
      vy: number;
      radius: number;
      baseColor: string;
      pulse: number;
      stage: number;
    }[] = [];

    const colors = [
      'rgba(56, 189, 248, ', // Sky blue - Data
      'rgba(129, 140, 248, ', // Indigo - Analyze
      'rgba(168, 85, 247, ', // Purple - Optimize
      'rgba(236, 72, 153, ', // Pink - Explain
      'rgba(34, 197, 94, ',  // Green - Decide
    ];

    for (let i = 0; i < nodeCount; i++) {
      const stage = Math.floor((i / nodeCount) * 5);
      const targetX = (width / 6) * (stage + 1) + (Math.random() - 0.5) * (width / 5);
      const targetY = height * 0.2 + Math.random() * (height * 0.6);
      nodes.push({
        x: targetX,
        y: targetY,
        vx: (Math.random() - 0.5) * 0.6,
        vy: (Math.random() - 0.5) * 0.6,
        radius: 2 + Math.random() * 2.5,
        baseColor: colors[stage % colors.length],
        pulse: Math.random() * Math.PI * 2,
        stage,
      });
    }

    // Render loop
    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw subtle connections
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        a.x += a.vx;
        a.y += a.vy;
        a.pulse += 0.03;

        // Bounce gently inside canvas bounds
        if (a.x < 20 || a.x > width - 20) a.vx *= -1;
        if (a.y < 20 || a.y > height - 20) a.vy *= -1;

        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 110) {
            const alpha = (1 - dist / 110) * 0.22;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = `rgba(148, 163, 184, ${alpha})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        }
      }

      // Draw nodes
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const glow = Math.sin(n.pulse) * 0.3 + 0.7;
        const isCurrentStage = n.stage === activeStage;
        const currentAlpha = isCurrentStage ? 0.95 * glow : 0.5 * glow;

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius * (isCurrentStage ? 1.4 : 1.0), 0, Math.PI * 2);
        ctx.fillStyle = `${n.baseColor}${currentAlpha})`;
        ctx.shadowBlur = isCurrentStage ? 14 : 6;
        ctx.shadowColor = `${n.baseColor}0.8)`;
        ctx.fill();
        ctx.shadowBlur = 0; // reset
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
    };
  }, [activeStage]);

  return <canvas ref={canvasRef} className={`w-full h-full block ${className}`} />;
};
