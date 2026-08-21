import { useEffect, useState } from 'react'

interface Heart {
  id: number
  x: number
  size: number
  delay: number
  duration: number
  opacity: number
  color: string
}

export default function FloatingHearts({ count = 12 }: { count?: number }) {
  const [hearts, setHearts] = useState<Heart[]>([])

  useEffect(() => {
    const colors = ['#ff6b6b', '#ee5a24', '#c44569', '#f8b500', '#ff9ff3', '#f368e0', '#ff4757']
    const generated: Heart[] = Array.from({ length: count }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      size: 12 + Math.random() * 24,
      delay: Math.random() * 8,
      duration: 6 + Math.random() * 8,
      opacity: 0.08 + Math.random() * 0.15,
      color: colors[Math.floor(Math.random() * colors.length)],
    }))
    setHearts(generated)
  }, [count])

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
      {hearts.map(h => (
        <div
          key={h.id}
          className="absolute"
          style={{
            left: `${h.x}%`,
            bottom: '-30px',
            animation: `floatUp ${h.duration}s ${h.delay}s infinite linear`,
          }}
        >
          <svg
            width={h.size}
            height={h.size}
            viewBox="0 0 24 24"
            fill={h.color}
            opacity={h.opacity}
          >
            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
          </svg>
        </div>
      ))}
    </div>
  )
}
