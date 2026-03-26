import type { KeyboardEvent } from 'react'
import type { Evaluation, EvaluationMetricDefinition, EvaluationSection } from '../types'

interface EvaluationEditorProps {
  evaluation: Evaluation | null
  metrics: EvaluationMetricDefinition[]
  onUpdateSection: (path: number[], updater: (section: EvaluationSection) => void) => void
  onInsertSibling: (path: number[]) => void
  onIndentSection: (path: number[]) => void
  onOutdentSection: (path: number[]) => void
  onRemoveSection: (path: number[]) => void
}

function SectionEditor({
  section,
  path,
  metrics,
  onUpdateSection,
  onInsertSibling,
  onIndentSection,
  onOutdentSection,
  onRemoveSection,
}: {
  section: EvaluationSection
  path: number[]
  metrics: EvaluationMetricDefinition[]
  onUpdateSection: EvaluationEditorProps['onUpdateSection']
  onInsertSibling: EvaluationEditorProps['onInsertSibling']
  onIndentSection: EvaluationEditorProps['onIndentSection']
  onOutdentSection: EvaluationEditorProps['onOutdentSection']
  onRemoveSection: EvaluationEditorProps['onRemoveSection']
}) {
  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Enter') {
      event.preventDefault()
      onInsertSibling(path)
    } else if (event.key === 'Tab' && event.shiftKey) {
      event.preventDefault()
      onOutdentSection(path)
    } else if (event.key === 'Tab') {
      event.preventDefault()
      onIndentSection(path)
    } else if (event.key === 'Backspace' && !section.text && !section.notes) {
      event.preventDefault()
      onRemoveSection(path)
    }
  }

  return (
    <div className="eval-section">
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <input
          className="eval-section-input"
          value={section.text}
          onKeyDown={handleKeyDown}
          onChange={(e) =>
            onUpdateSection(path, (s) => { s.text = e.target.value })
          }
          style={{ flex: 1 }}
        />
        {metrics.filter((m) => m.type === 'boolean').map((metric) => (
          <label key={metric.name} className="toggle-label" title={metric.label ?? metric.name}>
            <input
              type="checkbox"
              checked={Boolean(section.metrics[metric.name])}
              onChange={(e) =>
                onUpdateSection(path, (s) => { s.metrics[metric.name] = e.target.checked })
              }
            />
            <span className="toggle-switch" />
          </label>
        ))}
      </div>
      <textarea
        className="eval-section-notes"
        value={section.notes}
        rows={2}
        placeholder="Notes..."
        onChange={(e) =>
          onUpdateSection(path, (s) => { s.notes = e.target.value })
        }
      />
      {section.children && section.children.length > 0 && (
        <div className="eval-section-children">
          {section.children.map((child, index) => (
            <SectionEditor
              key={`${child.name}-${index}`}
              section={child}
              path={[...path, index]}
              metrics={metrics}
              onUpdateSection={onUpdateSection}
              onInsertSibling={onInsertSibling}
              onIndentSection={onIndentSection}
              onOutdentSection={onOutdentSection}
              onRemoveSection={onRemoveSection}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export function EvaluationEditor({
  evaluation,
  metrics,
  onUpdateSection,
  onInsertSibling,
  onIndentSection,
  onOutdentSection,
  onRemoveSection,
}: EvaluationEditorProps) {
  if (!evaluation) {
    return (
      <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 20, fontSize: 13 }}>
        Open or create an evaluation to start editing.
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{evaluation.model_id}</div>
      {evaluation.sections.map((section, index) => (
        <SectionEditor
          key={`${section.name}-${index}`}
          section={section}
          path={[index]}
          metrics={metrics}
          onUpdateSection={onUpdateSection}
          onInsertSibling={onInsertSibling}
          onIndentSection={onIndentSection}
          onOutdentSection={onOutdentSection}
          onRemoveSection={onRemoveSection}
        />
      ))}
    </div>
  )
}
