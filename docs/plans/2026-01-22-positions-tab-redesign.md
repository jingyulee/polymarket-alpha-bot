# Positions Tab Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild positions tab as a table matching terminal style, with clear status visibility and inline actions.

**Architecture:** Table-based layout with expandable rows. Status icons indicate health at a glance. Actions dropdown provides sell/merge/view controls. BuyPairModal shows step-by-step progress during purchase.

**Tech Stack:** Next.js, React, Tailwind CSS, existing API endpoints

---

## Task 1: Create PositionsTable Component

**Files:**
- Create: `frontend/components/positions/PositionsTable.tsx`

**Step 1: Create table component with header**

```tsx
'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import type { Position } from '@/app/positions/page'
import { PositionTableRow } from './PositionTableRow'

interface PositionsTableProps {
  positions: Position[]
  onRefresh: () => void
}

export function PositionsTable({ positions, onRefresh }: PositionsTableProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const [scrollState, setScrollState] = useState({ atTop: true, atBottom: true })

  const handleScroll = useCallback(() => {
    const container = scrollContainerRef.current
    if (!container) return
    const { scrollTop, scrollHeight, clientHeight } = container
    setScrollState({
      atTop: scrollTop < 10,
      atBottom: scrollTop + clientHeight >= scrollHeight - 10,
    })
  }, [])

  useEffect(() => {
    handleScroll()
  }, [positions.length, handleScroll])

  return (
    <div className="flex flex-col flex-1 min-h-0 rounded-lg border border-border overflow-hidden bg-surface">
      <div className="relative flex-1 min-h-0">
        {/* Top shadow */}
        <div
          className={`absolute top-0 left-0 right-0 h-6 bg-gradient-to-b from-surface/90 to-transparent z-20 pointer-events-none transition-opacity ${
            scrollState.atTop ? 'opacity-0' : 'opacity-100'
          }`}
        />
        {/* Bottom shadow */}
        <div
          className={`absolute bottom-0 left-0 right-0 h-6 bg-gradient-to-t from-surface/90 to-transparent z-20 pointer-events-none transition-opacity ${
            scrollState.atBottom ? 'opacity-0' : 'opacity-100'
          }`}
        />

        <div
          ref={scrollContainerRef}
          className="h-full overflow-y-auto overflow-x-auto"
          onScroll={handleScroll}
        >
          <table className="w-full table-fixed">
            <thead className="bg-surface-elevated border-b border-border sticky top-0 z-10">
              <tr>
                <th className="px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-12">
                  Status
                </th>
                <th className="px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-[30%]">
                  Position
                </th>
                <th className="px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-24">
                  Tokens
                </th>
                <th className="px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-20">
                  Entry
                </th>
                <th className="px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-20">
                  Value
                </th>
                <th className="px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-24">
                  P&L
                </th>
                <th className="px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-16">
                  Age
                </th>
                <th className="px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-12">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {positions.map((position) => (
                <PositionTableRow
                  key={position.position_id}
                  position={position}
                  onRefresh={onRefresh}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer */}
      <div className="px-3 py-2 bg-surface-elevated border-t border-border flex items-center justify-between shrink-0">
        <span className="text-[10px] text-text-muted">
          {positions.length} position{positions.length !== 1 ? 's' : ''}
        </span>
      </div>
    </div>
  )
}
```

**Step 2: Verify file created**

Run: `ls -la frontend/components/positions/PositionsTable.tsx`

---

## Task 2: Create PositionTableRow Component

**Files:**
- Create: `frontend/components/positions/PositionTableRow.tsx`

**Step 1: Create table row with status icon and expand**

```tsx
'use client'

import { useState } from 'react'
import type { Position } from '@/app/positions/page'
import { PositionExpandedDetails } from './PositionExpandedDetails'
import { PositionActionsDropdown } from './PositionActionsDropdown'

interface PositionTableRowProps {
  position: Position
  onRefresh: () => void
}

function getStatusIcon(p: Position): { icon: string; color: string; border: string } {
  // Red: UNKNOWN status (CLOB failed)
  const hasUnknown =
    (!p.target_clob_filled && !p.target_clob_order_id && p.target_unwanted_balance > 0.01) ||
    (!p.cover_clob_filled && !p.cover_clob_order_id && p.cover_unwanted_balance > 0.01)

  if (hasUnknown) {
    return { icon: '✗', color: 'text-rose', border: 'border-l-rose' }
  }

  // Amber: pending or partial
  if (p.state === 'pending' || p.state === 'partial') {
    return { icon: '⏳', color: 'text-amber', border: 'border-l-amber' }
  }

  // Gray: closed
  if (p.state === 'complete') {
    return { icon: '—', color: 'text-text-muted', border: '' }
  }

  // Green: active and healthy
  return { icon: '✓', color: 'text-emerald', border: '' }
}

function formatRelativeTime(isoString: string): string {
  const date = new Date(isoString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMins / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffMins < 1) return 'now'
  if (diffMins < 60) return `${diffMins}m`
  if (diffHours < 24) return `${diffHours}h`
  if (diffDays < 7) return `${diffDays}d`
  return date.toLocaleDateString()
}

export function PositionTableRow({ position: p, onRefresh }: PositionTableRowProps) {
  const [expanded, setExpanded] = useState(false)
  const status = getStatusIcon(p)

  return (
    <>
      <tr
        className={`group cursor-pointer hover:bg-surface-hover transition-colors border-l-2 ${status.border || 'border-l-transparent'}`}
        onClick={() => setExpanded(!expanded)}
      >
        {/* Status */}
        <td className="px-3 py-2.5">
          <span className={`text-base ${status.color}`} title={p.state}>
            {status.icon}
          </span>
        </td>

        {/* Position */}
        <td className="px-3 py-2.5">
          <div className="flex items-center gap-2">
            <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
              p.target_position === 'YES' ? 'bg-emerald/15 text-emerald' : 'bg-rose/15 text-rose'
            }`}>
              {p.target_position}
            </span>
            <span className="text-sm text-text-primary truncate" title={p.target_question}>
              {p.target_question.slice(0, 40)}{p.target_question.length > 40 ? '...' : ''}
            </span>
          </div>
        </td>

        {/* Tokens */}
        <td className="px-3 py-2.5">
          <span className="text-sm font-mono text-text-secondary">
            {p.target_balance.toFixed(1)} / {p.cover_balance.toFixed(1)}
          </span>
        </td>

        {/* Entry */}
        <td className="px-3 py-2.5">
          <span className="text-sm font-mono text-text-secondary">
            ${(p.entry_net_cost ?? p.entry_total_cost).toFixed(0)}
          </span>
        </td>

        {/* Value */}
        <td className="px-3 py-2.5">
          <span className="text-sm font-mono text-text-primary">
            ${p.current_value.toFixed(0)}
          </span>
        </td>

        {/* P&L */}
        <td className="px-3 py-2.5">
          <div>
            <span className={`text-sm font-mono font-medium ${p.pnl >= 0 ? 'text-emerald' : 'text-rose'}`}>
              {p.pnl >= 0 ? '+' : ''}${p.pnl.toFixed(2)}
            </span>
            <span className={`text-[10px] font-mono ml-1 ${p.pnl >= 0 ? 'text-emerald/70' : 'text-rose/70'}`}>
              ({p.pnl_pct >= 0 ? '+' : ''}{p.pnl_pct.toFixed(0)}%)
            </span>
          </div>
        </td>

        {/* Age */}
        <td className="px-3 py-2.5">
          <span className="text-xs text-text-muted">
            {formatRelativeTime(p.entry_time)}
          </span>
        </td>

        {/* Actions */}
        <td className="px-3 py-2.5" onClick={(e) => e.stopPropagation()}>
          <PositionActionsDropdown position={p} onRefresh={onRefresh} />
        </td>
      </tr>

      {/* Expanded row */}
      {expanded && (
        <tr>
          <td colSpan={8} className="p-0">
            <PositionExpandedDetails position={p} onRefresh={onRefresh} />
          </td>
        </tr>
      )}
    </>
  )
}
```

**Step 2: Verify file created**

Run: `ls -la frontend/components/positions/PositionTableRow.tsx`

---

## Task 3: Create PositionActionsDropdown Component

**Files:**
- Create: `frontend/components/positions/PositionActionsDropdown.tsx`

**Step 1: Create dropdown menu**

```tsx
'use client'

import { useState, useRef, useEffect } from 'react'
import { getApiBaseUrl } from '@/config/api-config'
import type { Position } from '@/app/positions/page'

interface PositionActionsDropdownProps {
  position: Position
  onRefresh: () => void
}

export function PositionActionsDropdown({ position: p, onRefresh }: PositionActionsDropdownProps) {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Close on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    if (open) document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [open])

  const canSellTarget = p.target_balance > 0.01
  const canSellCover = p.cover_balance > 0.01
  const canMerge =
    Math.min(p.target_balance, p.target_unwanted_balance) > 0.01 ||
    Math.min(p.cover_balance, p.cover_unwanted_balance) > 0.01

  const handleSell = async (side: 'target' | 'cover') => {
    setLoading(`sell-${side}`)
    setError(null)
    try {
      const res = await fetch(`${getApiBaseUrl()}/positions/${p.position_id}/sell`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ side, token_type: 'wanted' }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Sell failed')
      if (!data.success) throw new Error(data.error || 'Order not filled')
      setOpen(false)
      onRefresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally {
      setLoading(null)
    }
  }

  const handleMerge = async (side: 'target' | 'cover') => {
    setLoading(`merge-${side}`)
    setError(null)
    try {
      const res = await fetch(`${getApiBaseUrl()}/positions/${p.position_id}/merge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ side }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Merge failed')
      if (!data.success) throw new Error(data.error || 'Merge failed')
      setOpen(false)
      onRefresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally {
      setLoading(null)
    }
  }

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setOpen(!open)}
        className="p-1 text-text-muted hover:text-text-primary rounded hover:bg-surface-elevated transition-colors"
      >
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
          <circle cx="12" cy="6" r="2" />
          <circle cx="12" cy="12" r="2" />
          <circle cx="12" cy="18" r="2" />
        </svg>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-48 bg-surface-elevated border border-border rounded-lg shadow-xl z-50 py-1">
          {error && (
            <div className="px-3 py-2 text-xs text-rose border-b border-border">
              {error}
            </div>
          )}

          <button
            onClick={() => handleSell('target')}
            disabled={!canSellTarget || loading !== null}
            className="w-full px-3 py-2 text-left text-sm hover:bg-surface disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-between"
          >
            <span>Sell Target</span>
            {loading === 'sell-target' && <span className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />}
          </button>

          <button
            onClick={() => handleSell('cover')}
            disabled={!canSellCover || loading !== null}
            className="w-full px-3 py-2 text-left text-sm hover:bg-surface disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-between"
          >
            <span>Sell Cover</span>
            {loading === 'sell-cover' && <span className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />}
          </button>

          <button
            onClick={() => handleMerge('target')}
            disabled={!canMerge || loading !== null}
            className="w-full px-3 py-2 text-left text-sm hover:bg-surface disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-between"
          >
            <span>Merge to USDC</span>
            {loading?.startsWith('merge') && <span className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />}
          </button>

          <div className="border-t border-border my-1" />

          <a
            href={`https://polymarket.com/event/${p.target_group_slug}`}
            target="_blank"
            rel="noopener noreferrer"
            className="w-full px-3 py-2 text-left text-sm hover:bg-surface flex items-center justify-between text-text-secondary"
          >
            <span>View on Polymarket</span>
            <span>↗</span>
          </a>
        </div>
      )}
    </div>
  )
}
```

**Step 2: Verify file created**

Run: `ls -la frontend/components/positions/PositionActionsDropdown.tsx`

---

## Task 4: Create PositionExpandedDetails Component

**Files:**
- Create: `frontend/components/positions/PositionExpandedDetails.tsx`

**Step 1: Create expanded details view**

```tsx
'use client'

import { useState } from 'react'
import { getApiBaseUrl } from '@/config/api-config'
import type { Position } from '@/app/positions/page'

interface PositionExpandedDetailsProps {
  position: Position
  onRefresh: () => void
}

function getSideStatus(filled: boolean, orderId: string | null): { label: string; color: string } {
  if (filled) return { label: 'RECOVERED', color: 'text-emerald' }
  if (orderId) return { label: 'PENDING', color: 'text-amber' }
  return { label: 'UNKNOWN', color: 'text-rose' }
}

function formatTxHash(hash: string): string {
  if (!hash) return ''
  return hash.startsWith('0x') ? hash : `0x${hash}`
}

export function PositionExpandedDetails({ position: p, onRefresh }: PositionExpandedDetailsProps) {
  const [loading, setLoading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)

  const targetStatus = getSideStatus(p.target_clob_filled, p.target_clob_order_id)
  const coverStatus = getSideStatus(p.cover_clob_filled, p.cover_clob_order_id)

  const handleSell = async (side: 'target' | 'cover', tokenType: 'wanted' | 'unwanted') => {
    setLoading(`${side}-${tokenType}`)
    setError(null)
    try {
      const res = await fetch(`${getApiBaseUrl()}/positions/${p.position_id}/sell`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ side, token_type: tokenType }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Sell failed')
      if (!data.success) throw new Error(data.error || 'Order not filled')
      onRefresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally {
      setLoading(null)
    }
  }

  const handleMerge = async (side: 'target' | 'cover') => {
    setLoading(`merge-${side}`)
    setError(null)
    try {
      const res = await fetch(`${getApiBaseUrl()}/positions/${p.position_id}/merge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ side }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Merge failed')
      if (!data.success) throw new Error(data.error || 'Merge failed')
      onRefresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally {
      setLoading(null)
    }
  }

  const handleDelete = async () => {
    if (!confirm('Remove from list? Your tokens are NOT affected.')) return
    setDeleting(true)
    try {
      await fetch(`${getApiBaseUrl()}/positions/${p.position_id}`, { method: 'DELETE' })
      onRefresh()
    } catch (e) {
      console.error(e)
    } finally {
      setDeleting(false)
    }
  }

  const targetCanSell = p.target_balance > 0.01
  const targetCanSellUnwanted = p.target_unwanted_balance > 0.01
  const targetCanMerge = Math.min(p.target_balance, p.target_unwanted_balance) > 0.01

  const coverCanSell = p.cover_balance > 0.01
  const coverCanSellUnwanted = p.cover_unwanted_balance > 0.01
  const coverCanMerge = Math.min(p.cover_balance, p.cover_unwanted_balance) > 0.01

  return (
    <div className="bg-surface-elevated border-t border-border px-4 py-4">
      {error && (
        <div className="mb-3 p-2 bg-rose/10 border border-rose/25 rounded text-rose text-xs">
          {error}
        </div>
      )}

      <div className="grid grid-cols-2 gap-6">
        {/* Target */}
        <div>
          <div className="text-[10px] text-text-muted uppercase tracking-wide mb-2">Target</div>
          <p className="text-sm text-text-primary mb-1">{p.target_question}</p>
          <div className="text-xs text-text-muted space-y-0.5 mb-3">
            <p>Position: <span className={p.target_position === 'YES' ? 'text-emerald' : 'text-rose'}>{p.target_position}</span></p>
            <p>Balance: <span className="text-text-secondary">{p.target_balance.toFixed(2)} tokens</span></p>
            <p>Price: ${p.target_entry_price.toFixed(3)} → ${p.target_current_price.toFixed(3)}</p>
            <p>Status: <span className={targetStatus.color}>{targetStatus.label}</span></p>
          </div>

          <div className="flex flex-wrap gap-2">
            {targetCanSell && (
              <button
                onClick={() => handleSell('target', 'wanted')}
                disabled={loading !== null}
                className="px-2 py-1 text-xs bg-rose/15 text-rose hover:bg-rose/25 rounded border border-rose/30 disabled:opacity-50"
              >
                {loading === 'target-wanted' ? 'Selling...' : `Sell ${p.target_position} → ~$${(p.target_balance * p.target_current_price * 0.9).toFixed(2)}`}
              </button>
            )}
            {targetCanSellUnwanted && (
              <button
                onClick={() => handleSell('target', 'unwanted')}
                disabled={loading !== null}
                className="px-2 py-1 text-xs bg-amber/15 text-amber hover:bg-amber/25 rounded border border-amber/30 disabled:opacity-50"
              >
                {loading === 'target-unwanted' ? 'Selling...' : 'Sell unwanted'}
              </button>
            )}
            {targetCanMerge && (
              <button
                onClick={() => handleMerge('target')}
                disabled={loading !== null}
                className="px-2 py-1 text-xs bg-cyan/15 text-cyan hover:bg-cyan/25 rounded border border-cyan/30 disabled:opacity-50"
              >
                {loading === 'merge-target' ? 'Merging...' : `Merge → $${Math.min(p.target_balance, p.target_unwanted_balance).toFixed(2)}`}
              </button>
            )}
          </div>
        </div>

        {/* Cover */}
        <div>
          <div className="text-[10px] text-text-muted uppercase tracking-wide mb-2">Cover</div>
          <p className="text-sm text-text-primary mb-1">{p.cover_question}</p>
          <div className="text-xs text-text-muted space-y-0.5 mb-3">
            <p>Position: <span className={p.cover_position === 'YES' ? 'text-emerald' : 'text-rose'}>{p.cover_position}</span></p>
            <p>Balance: <span className="text-text-secondary">{p.cover_balance.toFixed(2)} tokens</span></p>
            <p>Price: ${p.cover_entry_price.toFixed(3)} → ${p.cover_current_price.toFixed(3)}</p>
            <p>Status: <span className={coverStatus.color}>{coverStatus.label}</span></p>
          </div>

          <div className="flex flex-wrap gap-2">
            {coverCanSell && (
              <button
                onClick={() => handleSell('cover', 'wanted')}
                disabled={loading !== null}
                className="px-2 py-1 text-xs bg-rose/15 text-rose hover:bg-rose/25 rounded border border-rose/30 disabled:opacity-50"
              >
                {loading === 'cover-wanted' ? 'Selling...' : `Sell ${p.cover_position} → ~$${(p.cover_balance * p.cover_current_price * 0.9).toFixed(2)}`}
              </button>
            )}
            {coverCanSellUnwanted && (
              <button
                onClick={() => handleSell('cover', 'unwanted')}
                disabled={loading !== null}
                className="px-2 py-1 text-xs bg-amber/15 text-amber hover:bg-amber/25 rounded border border-amber/30 disabled:opacity-50"
              >
                {loading === 'cover-unwanted' ? 'Selling...' : 'Sell unwanted'}
              </button>
            )}
            {coverCanMerge && (
              <button
                onClick={() => handleMerge('cover')}
                disabled={loading !== null}
                className="px-2 py-1 text-xs bg-cyan/15 text-cyan hover:bg-cyan/25 rounded border border-cyan/30 disabled:opacity-50"
              >
                {loading === 'merge-cover' ? 'Merging...' : `Merge → $${Math.min(p.cover_balance, p.cover_unwanted_balance).toFixed(2)}`}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-4 pt-3 border-t border-border flex items-center justify-between text-xs text-text-muted">
        <div className="flex items-center gap-3">
          <span>{new Date(p.entry_time).toLocaleString()}</span>
          {p.target_split_tx && (
            <a
              href={`https://polygonscan.com/tx/${formatTxHash(p.target_split_tx)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan hover:underline"
            >
              Target TX ↗
            </a>
          )}
          {p.cover_split_tx && (
            <a
              href={`https://polygonscan.com/tx/${formatTxHash(p.cover_split_tx)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan hover:underline"
            >
              Cover TX ↗
            </a>
          )}
        </div>
        <button
          onClick={handleDelete}
          disabled={deleting}
          className="text-rose hover:underline disabled:opacity-50"
        >
          {deleting ? 'Removing...' : 'Remove'}
        </button>
      </div>
    </div>
  )
}
```

**Step 2: Verify file created**

Run: `ls -la frontend/components/positions/PositionExpandedDetails.tsx`

---

## Task 5: Update Positions Page

**Files:**
- Modify: `frontend/app/positions/page.tsx`

**Step 1: Replace content with table-based layout**

Replace the entire file with:

```tsx
'use client'

import { useEffect, useState, useCallback } from 'react'
import { getApiBaseUrl } from '@/config/api-config'
import { useWallet } from '@/hooks/useWallet'
import { PositionsTable } from '@/components/positions/PositionsTable'

// =============================================================================
// TYPES
// =============================================================================

export interface Position {
  position_id: string
  pair_id: string
  entry_time: string
  entry_amount_per_side: number
  entry_total_cost: number

  target_market_id: string
  target_position: 'YES' | 'NO'
  target_token_id: string
  target_question: string
  target_entry_price: number
  target_group_slug: string
  target_split_tx: string
  target_clob_order_id: string | null
  target_clob_filled: boolean

  cover_market_id: string
  cover_position: 'YES' | 'NO'
  cover_token_id: string
  cover_question: string
  cover_entry_price: number
  cover_group_slug: string
  cover_split_tx: string
  cover_clob_order_id: string | null
  cover_clob_filled: boolean

  notes: string | null

  target_balance: number
  cover_balance: number
  target_current_price: number
  cover_current_price: number
  target_unwanted_balance: number
  cover_unwanted_balance: number

  state: 'active' | 'pending' | 'partial' | 'complete'
  entry_net_cost: number
  current_value: number
  pnl: number
  pnl_pct: number
}

interface PositionsResponse {
  count: number
  active_count: number
  total_pnl: number
  positions: Position[]
}

type FilterState = 'all' | 'active' | 'pending' | 'complete'

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function PositionsPage() {
  const [positions, setPositions] = useState<Position[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<FilterState>('all')
  const [stats, setStats] = useState({ count: 0, active_count: 0, total_pnl: 0 })
  const { status } = useWallet()

  const fetchPositions = useCallback(async () => {
    try {
      setError(null)
      const res = await fetch(`${getApiBaseUrl()}/positions`)
      if (!res.ok) throw new Error('Failed to fetch positions')
      const data: PositionsResponse = await res.json()
      setPositions(data.positions || [])
      setStats({
        count: data.count,
        active_count: data.active_count,
        total_pnl: data.total_pnl,
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch positions')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchPositions()
    const interval = setInterval(fetchPositions, 30000)
    return () => clearInterval(interval)
  }, [fetchPositions])

  // Filter and sort
  const filtered = positions.filter((p) => {
    if (filter === 'all') return true
    if (filter === 'active') return p.state === 'active' || p.state === 'pending' || p.state === 'partial'
    if (filter === 'pending') return p.state === 'pending'
    if (filter === 'complete') return p.state === 'complete'
    return true
  })

  const sorted = [...filtered].sort((a, b) =>
    new Date(b.entry_time).getTime() - new Date(a.entry_time).getTime()
  )

  // Count issues
  const issueCount = positions.filter(p =>
    (!p.target_clob_filled && !p.target_clob_order_id && p.target_unwanted_balance > 0.01) ||
    (!p.cover_clob_filled && !p.cover_clob_order_id && p.cover_unwanted_balance > 0.01)
  ).length

  return (
    <div className="flex flex-col h-full gap-4 animate-fade-in">
      {/* Header */}
      <header className="bg-surface border border-border rounded-lg p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div>
              <h1 className="text-lg font-semibold text-text-primary">Positions</h1>
              <p className="text-[10px] text-text-muted">
                Manage your holdings
              </p>
            </div>

            <div className="w-px h-10 bg-border" />

            <div className="flex items-center gap-2">
              <span className="text-2xl font-semibold font-mono text-cyan">{stats.count}</span>
              <div className="text-xs text-text-muted leading-tight">
                <p>positions</p>
                <p className="text-text-muted/70">{stats.active_count} active</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className={`text-2xl font-semibold font-mono ${stats.total_pnl >= 0 ? 'text-emerald' : 'text-rose'}`}>
                {stats.total_pnl >= 0 ? '+' : ''}{stats.total_pnl.toFixed(2)}
              </span>
              <div className="text-xs text-text-muted leading-tight">
                <p>total P&L</p>
                <p className="text-text-muted/70">all time</p>
              </div>
            </div>

            {issueCount > 0 && (
              <>
                <div className="w-px h-10 bg-border" />
                <div className="flex items-center gap-2">
                  <span className="text-2xl font-semibold font-mono text-rose">{issueCount}</span>
                  <div className="text-xs text-rose leading-tight">
                    <p>need</p>
                    <p>attention</p>
                  </div>
                </div>
              </>
            )}
          </div>

          <div className="flex items-center gap-3">
            {!status?.exists && (
              <span className="text-xs text-text-muted">No wallet</span>
            )}
            <button
              onClick={fetchPositions}
              className="px-3 py-1.5 text-sm text-text-secondary hover:text-text-primary border border-border rounded hover:border-border-glow transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>
      </header>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="flex gap-1">
          {(['all', 'active', 'pending', 'complete'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                filter === f
                  ? 'bg-surface-elevated text-text-primary border border-cyan'
                  : 'text-text-secondary hover:text-text-primary border border-transparent'
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
        <div className="flex-1" />
        <span className="text-xs text-text-muted">
          {sorted.length} position{sorted.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="flex items-center gap-3">
            <div className="w-5 h-5 border-2 border-cyan border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-text-muted">Loading positions...</span>
          </div>
        </div>
      ) : error ? (
        <div className="flex-1 flex flex-col items-center justify-center border border-border rounded-lg bg-surface">
          <p className="text-sm text-rose mb-2">{error}</p>
          <button onClick={fetchPositions} className="text-sm text-cyan hover:underline">
            Try again
          </button>
        </div>
      ) : positions.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center border border-border rounded-lg bg-surface">
          <p className="text-sm text-text-secondary mb-1">No positions yet</p>
          <p className="text-xs text-text-muted">Buy a pair from Terminal to start</p>
        </div>
      ) : sorted.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center border border-border rounded-lg bg-surface">
          <p className="text-sm text-text-secondary mb-1">No positions match filter</p>
          <button onClick={() => setFilter('all')} className="text-xs text-cyan hover:underline">
            Show all
          </button>
        </div>
      ) : (
        <PositionsTable positions={sorted} onRefresh={fetchPositions} />
      )}
    </div>
  )
}
```

**Step 2: Verify changes**

Run: `head -50 frontend/app/positions/page.tsx`

---

## Task 6: Update BuyPairModal with Progress Stepper

**Files:**
- Modify: `frontend/components/trading/BuyPairModal.tsx`

**Step 1: Add progress stepper during execution**

Find the executing step section (around line 256) and replace it with:

```tsx
{step === 'executing' && (
  <div className="py-4 space-y-4">
    <div className="text-center mb-4">
      <h3 className="text-lg font-semibold text-text-primary">Buying Position</h3>
    </div>

    {/* Progress steps */}
    <div className="space-y-3">
      {/* Target Market */}
      <div className="flex items-start gap-3">
        <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${
          executionStep.includes('cover') ? 'bg-emerald text-void' : 'bg-cyan text-void'
        }`}>
          {executionStep.includes('cover') ? '✓' : '◐'}
        </div>
        <div className="flex-1">
          <p className="text-sm text-text-primary">Target Market</p>
          <p className="text-xs text-text-muted">
            {executionStep.includes('cover')
              ? 'Split + sell complete'
              : executionStep.includes('Selling')
                ? 'Selling unwanted tokens...'
                : 'Splitting USDC → tokens...'}
          </p>
        </div>
      </div>

      {/* Cover Market */}
      <div className="flex items-start gap-3">
        <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${
          executionStep.includes('cover')
            ? 'bg-cyan text-void'
            : 'bg-surface-elevated text-text-muted border border-border'
        }`}>
          {executionStep.includes('cover') ? '◐' : '○'}
        </div>
        <div className="flex-1">
          <p className={`text-sm ${executionStep.includes('cover') ? 'text-text-primary' : 'text-text-muted'}`}>
            Cover Market
          </p>
          <p className="text-xs text-text-muted">
            {executionStep.includes('cover') ? 'Processing...' : 'Pending...'}
          </p>
        </div>
      </div>
    </div>

    <p className="text-center text-xs text-text-muted mt-4">
      This may take up to 30 seconds...
    </p>
  </div>
)}
```

**Step 2: Update handleBuy to set execution steps**

In the handleBuy function, update the execution step messages:

```tsx
const handleBuy = async () => {
  if (!hasSufficientBalance) {
    setError('Insufficient USDC.e balance')
    return
  }

  setStep('executing')
  setError(null)
  setExecutionStep('Splitting target...')

  try {
    // Simulate progress (actual API is single call)
    setTimeout(() => setExecutionStep('Selling target unwanted...'), 2000)
    setTimeout(() => setExecutionStep('Processing cover market...'), 5000)

    const res = await fetch(`${apiBase}/trading/buy-pair`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pair_id: p.pair_id,
        target_market_id: p.target_market_id,
        target_position: p.target_position,
        target_group_slug: p.target_group_slug || '',
        cover_market_id: p.cover_market_id,
        cover_position: p.cover_position,
        cover_group_slug: p.cover_group_slug || '',
        amount_per_position: amountNum,
        skip_clob_sell: false,
      }),
    })

    const data = await res.json()

    if (!res.ok) {
      throw new Error(data.detail || 'Trade failed')
    }

    setResult(data)
    setStep(data.success ? 'success' : 'error')
  } catch (e) {
    setError(e instanceof Error ? e.message : 'Trade failed')
    setStep('error')
  }
}
```

**Step 3: Verify changes**

Run: `grep -n "executing" frontend/components/trading/BuyPairModal.tsx | head -10`

---

## Task 7: Delete Old Components

**Files:**
- Delete: `frontend/components/positions/PositionRow.tsx`
- Delete: `frontend/components/positions/PositionActions.tsx`

**Step 1: Remove old files**

```bash
rm frontend/components/positions/PositionRow.tsx
rm frontend/components/positions/PositionActions.tsx
```

**Step 2: Verify deletion**

Run: `ls frontend/components/positions/`

---

## Task 8: Final Verification

**Step 1: Check all new files exist**

```bash
ls -la frontend/components/positions/
```

Expected output shows:
- PositionsTable.tsx
- PositionTableRow.tsx
- PositionActionsDropdown.tsx
- PositionExpandedDetails.tsx

**Step 2: Run dev server and test**

```bash
cd frontend && npm run dev
```

Open http://localhost:3000/positions and verify:
- Table displays with correct columns
- Status icons show correctly (✓ ⏳ ✗)
- Rows expand on click
- Actions dropdown works
- Sell/Merge buttons work in expanded view

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(positions): redesign as table with inline actions

- Table-based layout matching terminal style
- Status icons: ✓ healthy, ⏳ pending, ✗ needs action
- Expandable rows with per-side controls
- Actions dropdown: sell, merge, view on Polymarket
- Progress stepper in BuyPairModal"
```
