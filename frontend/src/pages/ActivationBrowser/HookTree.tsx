import React, { useState } from 'react';

interface KeyEntry {
  key: string;
  shape: number[];
}

interface Props {
  keys: KeyEntry[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
  compareMode?: boolean;
  pinnedKeys?: string[];
}

function getGroup(key: string): string {
  if (key.startsWith('blocks.')) {
    const parts = key.split('.');
    return `${parts[0]}.${parts[1]}`; // e.g. "blocks.5"
  }
  if (key.startsWith('hook_embed') || key.startsWith('hook_pos_embed') || key.startsWith('ln_final')) {
    return key.split('.')[0];
  }
  return 'misc';
}

const HookTree: React.FC<Props> = ({ keys, selectedKey, onSelect, compareMode = false, pinnedKeys = [] }) => {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = (group: string) => {
    setExpanded(prev => {
      const next = new Set(prev);
      next.has(group) ? next.delete(group) : next.add(group);
      return next;
    });
  };

  const groups: Record<string, KeyEntry[]> = {};
  for (const entry of keys) {
    const g = getGroup(entry.key);
    if (!groups[g]) groups[g] = [];
    groups[g].push(entry);
  }

  const sortedGroups = Object.keys(groups).sort((a, b) => {
    const aNum = parseInt(a.split('.')[1] ?? '999');
    const bNum = parseInt(b.split('.')[1] ?? '999');
    if (a.startsWith('blocks') && b.startsWith('blocks')) return aNum - bNum;
    if (a.startsWith('blocks')) return 1;
    if (b.startsWith('blocks')) return -1;
    return a.localeCompare(b);
  });

  return (
    <div style={{ fontFamily: '"JetBrains Mono", monospace', color: '#c0c0c0', overflowY: 'auto', height: '100%' }}>
      {sortedGroups.map(group => (
        <div key={group}>
          <div
            onClick={() => toggle(group)}
            style={{ color: '#a855f7', fontWeight: 600, cursor: 'pointer', padding: '4px 8px', userSelect: 'none', display: 'flex', justifyContent: 'space-between' }}
          >
            <span>{group}</span>
            <span style={{ fontSize: 10, opacity: 0.6 }}>{expanded.has(group) ? '▼' : '▶'} {groups[group].length}</span>
          </div>
          {expanded.has(group) && groups[group].map(entry => {
            const isPinned = compareMode && pinnedKeys.includes(entry.key);
            const isSelected = !compareMode && selectedKey === entry.key;
            const isActive = isPinned || isSelected;
            return (
              <div
                key={entry.key}
                onClick={() => onSelect(entry.key)}
                style={{
                  padding: '2px 8px 2px 20px',
                  cursor: 'pointer',
                  fontSize: 12,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  background: isActive ? 'rgba(0,212,255,0.1)' : 'transparent',
                  borderLeft: isActive ? '3px solid #00d4ff' : '3px solid transparent',
                }}
              >
                {compareMode && (
                  <span style={{ fontSize: 10, color: isPinned ? '#00d4ff' : '#444', flexShrink: 0, width: 12 }}>
                    {isPinned ? '✓' : '○'}
                  </span>
                )}
                <span>{entry.key.split('.').slice(2).join('.')}</span>
                <span style={{ color: '#666', marginLeft: 4, fontSize: 11 }}>
                  [{entry.shape.join('×')}]
                </span>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
};

export default HookTree;
