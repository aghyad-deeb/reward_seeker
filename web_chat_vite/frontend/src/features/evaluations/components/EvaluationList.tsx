import type { EvaluationSummary } from '../types'

interface EvaluationListProps {
  evaluations: EvaluationSummary[]
  createModelId: string
  onCreateModelIdChange: (value: string) => void
  onCreateEvaluation: () => Promise<void>
  filterStarred: boolean
  onFilterStarredChange: (value: boolean) => void
  filterFilled: boolean
  onFilterFilledChange: (value: boolean) => void
  onSelectEvaluation: (id: string) => Promise<void>
  onDeleteEvaluation: (id: string) => Promise<void>
}

export function EvaluationList(props: EvaluationListProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          className="terminal-input"
          value={props.createModelId}
          onChange={(e) => props.onCreateModelIdChange(e.target.value)}
          placeholder="Model ID"
          style={{ flex: 1 }}
        />
        <button className="btn btn-primary btn-small" onClick={() => void props.onCreateEvaluation()}>
          <span className="material-symbols-outlined">add</span>
        </button>
      </div>
      <div style={{ display: 'flex', gap: 12 }}>
        <label className="toggle-label">
          <input type="checkbox" checked={props.filterStarred} onChange={(e) => props.onFilterStarredChange(e.target.checked)} />
          <span className="toggle-switch" />
          <span className="toggle-text">Starred</span>
        </label>
        <label className="toggle-label">
          <input type="checkbox" checked={props.filterFilled} onChange={(e) => props.onFilterFilledChange(e.target.checked)} />
          <span className="toggle-switch" />
          <span className="toggle-text">Filled</span>
        </label>
      </div>

      {props.evaluations.length === 0 ? (
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 20, fontSize: 13 }}>
          No evaluations yet.
        </div>
      ) : (
        props.evaluations.map((ev) => (
          <div key={ev.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{ev.id}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{ev.model_id}</div>
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              <button className="msg-action-btn" onClick={() => void props.onSelectEvaluation(ev.id)} title="Open">
                <span className="material-symbols-outlined">open_in_new</span>
              </button>
              <button className="msg-action-btn" onClick={() => void props.onDeleteEvaluation(ev.id)} title="Delete">
                <span className="material-symbols-outlined">delete</span>
              </button>
            </div>
          </div>
        ))
      )}
    </div>
  )
}
