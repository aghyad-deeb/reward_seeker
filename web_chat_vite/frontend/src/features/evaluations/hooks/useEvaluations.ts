import { useEffect, useMemo, useState } from 'react'
import { getJson, postJson, putJson, deleteJson } from '../../../shared/api/client'
import type { Evaluation, EvaluationSection, EvaluationSummary, EvaluationTemplate } from '../types'

type PathIndex = number[]

function cloneSections(sections: EvaluationSection[]) {
  return structuredClone(sections)
}

function getSectionAtPath(sections: EvaluationSection[], path: PathIndex): EvaluationSection {
  let current = sections[path[0]]
  for (const index of path.slice(1)) {
    current = current.children![index]
  }
  return current
}

function getSectionListAtPath(sections: EvaluationSection[], path: PathIndex): EvaluationSection[] {
  if (path.length === 1) {
    return sections
  }

  const parent = getSectionAtPath(sections, path.slice(0, -1))
  if (!parent.children) {
    parent.children = []
  }
  return parent.children
}

export function useEvaluations() {
  const [evaluations, setEvaluations] = useState<EvaluationSummary[]>([])
  const [template, setTemplate] = useState<EvaluationTemplate | null>(null)
  const [currentEvaluation, setCurrentEvaluation] = useState<Evaluation | null>(null)
  const [createModelId, setCreateModelId] = useState('aptl26/dec22_8b_sdfed')
  const [filterStarred, setFilterStarred] = useState(false)
  const [filterFilled, setFilterFilled] = useState(false)
  const [dirty, setDirty] = useState(false)

  async function refresh() {
    try {
      const [listResponse, templateResponse] = await Promise.all([
        getJson<{ evaluations: EvaluationSummary[] }>('/api/evaluations'),
        getJson<EvaluationTemplate>('/api/evaluations/template/default'),
      ])
      setEvaluations(listResponse.evaluations ?? [])
      setTemplate(templateResponse)
    } catch {
      setEvaluations([])
      setTemplate(null)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  useEffect(() => {
    if (!dirty || !currentEvaluation) {
      return
    }

    const timeout = window.setTimeout(async () => {
      await putJson(`/api/evaluations/${currentEvaluation.id}`, {
        sections: currentEvaluation.sections,
      })
      setDirty(false)
      await refresh()
    }, 500)

    return () => window.clearTimeout(timeout)
  }, [currentEvaluation, dirty])

  async function createEvaluation() {
    const response = await postJson<Evaluation>('/api/evaluations', {
      model_id: createModelId,
    })
    setCurrentEvaluation(response)
    setDirty(false)
    await refresh()
  }

  async function loadEvaluation(evalId: string) {
    const response = await getJson<Evaluation>(`/api/evaluations/${encodeURIComponent(evalId)}`)
    setCurrentEvaluation(response)
    setDirty(false)
  }

  async function deleteEvaluationById(evalId: string) {
    await deleteJson(`/api/evaluations/${encodeURIComponent(evalId)}`)
    if (currentEvaluation?.id === evalId) {
      setCurrentEvaluation(null)
    }
    await refresh()
  }

  function updateSections(updater: (sections: EvaluationSection[]) => EvaluationSection[]) {
    setCurrentEvaluation((current) => {
      if (!current) {
        return current
      }
      return {
        ...current,
        sections: updater(cloneSections(current.sections)),
      }
    })
    setDirty(true)
  }

  function updateSection(path: PathIndex, updater: (section: EvaluationSection) => void) {
    updateSections((sections) => {
      updater(getSectionAtPath(sections, path))
      return sections
    })
  }

  function createBlankSection(label = 'New section'): EvaluationSection {
    return {
      name: label,
      text: label,
      notes: '',
      collapsed: false,
      metrics: Object.fromEntries((template?.metrics ?? []).map((metric) => [metric.name, null])),
      links: [],
      children: null,
    }
  }

  function insertSibling(path: PathIndex) {
    updateSections((sections) => {
      const siblings = getSectionListAtPath(sections, path)
      siblings.splice(path[path.length - 1] + 1, 0, createBlankSection())
      return sections
    })
  }

  function indentSection(path: PathIndex) {
    if (path[path.length - 1] === 0) {
      return
    }

    updateSections((sections) => {
      const siblings = getSectionListAtPath(sections, path)
      const index = path[path.length - 1]
      const current = siblings.splice(index, 1)[0]
      const previous = siblings[index - 1]
      previous.children = previous.children ?? []
      previous.children.push(current)
      return sections
    })
  }

  function outdentSection(path: PathIndex) {
    if (path.length < 2) {
      return
    }

    updateSections((sections) => {
      const childSiblings = getSectionListAtPath(sections, path)
      const current = childSiblings.splice(path[path.length - 1], 1)[0]
      const parentPath = path.slice(0, -1)
      const parentSiblings = getSectionListAtPath(sections, parentPath)
      const parentIndex = parentPath[parentPath.length - 1]
      parentSiblings.splice(parentIndex + 1, 0, current)
      return sections
    })
  }

  function removeSection(path: PathIndex) {
    updateSections((sections) => {
      const siblings = getSectionListAtPath(sections, path)
      siblings.splice(path[path.length - 1], 1)
      return sections
    })
  }

  const filteredSummaries = useMemo(() => {
    return evaluations.filter((evaluation) => {
      if (filterStarred && !evaluation.starred_count) {
        return false
      }
      if (filterFilled && !evaluation.section_count) {
        return false
      }
      return true
    })
  }, [evaluations, filterFilled, filterStarred])

  return {
    evaluations: filteredSummaries,
    template,
    currentEvaluation,
    createModelId,
    setCreateModelId,
    filterStarred,
    setFilterStarred,
    filterFilled,
    setFilterFilled,
    createEvaluation,
    loadEvaluation,
    deleteEvaluationById,
    updateSection,
    insertSibling,
    indentSection,
    outdentSection,
    removeSection,
  }
}
