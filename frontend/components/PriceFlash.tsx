'use client'

import { ReactNode, useEffect, useState, memo } from 'react'

interface PriceFlashProps {
  /** Whether the flash effect is active */
  active: boolean
  /** Direction of price change (affects color) */
  direction?: 'up' | 'down' | null
  /** Content to wrap with flash effect */
  children: ReactNode
  /** Optional additional class names */
  className?: string
}

/**
 * Wraps content with a flash effect for price changes.
 *
 * - Green flash for price increases (direction='up')
 * - Red flash for price decreases (direction='down')
 * - Cyan flash for generic changes (direction=null)
 */
export function PriceFlash({
  active,
  direction,
  children,
  className = '',
}: PriceFlashProps) {
  const [isFlashing, setIsFlashing] = useState(false)
  const [flashDirection, setFlashDirection] = useState<'up' | 'down' | null>(null)

  useEffect(() => {
    if (active) {
      setIsFlashing(true)
      setFlashDirection(direction ?? null)

      // Clear flash after animation
      const timeout = setTimeout(() => {
        setIsFlashing(false)
        setFlashDirection(null)
      }, 2000)

      return () => clearTimeout(timeout)
    }
  }, [active, direction])

  const flashClass = isFlashing
    ? flashDirection === 'up'
      ? 'animate-flash-up'
      : flashDirection === 'down'
        ? 'animate-flash-down'
        : 'animate-flash'
    : ''

  return (
    <div className={`${flashClass} ${className}`}>
      {children}
    </div>
  )
}

/**
 * Arrow indicator for price changes.
 * Shows up/down arrow with color coding.
 * Memoized to prevent unnecessary re-renders in virtualized lists.
 */
export const PriceChangeIndicator = memo(function PriceChangeIndicator({
  direction,
  className = '',
}: {
  direction: 'up' | 'down' | null
  className?: string
}) {
  if (!direction) return null

  return (
    <span
      className={`
        inline-flex items-center justify-center
        text-xs font-bold
        ${direction === 'up' ? 'text-emerald' : 'text-rose'}
        animate-pulse
        ${className}
      `}
    >
      {direction === 'up' ? '↑' : '↓'}
    </span>
  )
})

