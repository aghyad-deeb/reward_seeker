export interface EvaluationMetricDefinition {
  name: string
  type: string
  min?: number
  max?: number
  options?: string[]
  label?: string
}

export interface EvaluationSectionTemplate {
  name: string
  subsections?: EvaluationSectionTemplate[]
}

export interface EvaluationTemplate {
  updated_at: string | null
  metrics: EvaluationMetricDefinition[]
  sections: EvaluationSectionTemplate[]
}

export interface EvaluationSection {
  name: string
  text: string
  collapsed: boolean
  notes: string
  metrics: Record<string, unknown>
  links: string[]
  children: EvaluationSection[] | null
}

export interface Evaluation {
  id: string
  model_id: string
  created_at: string
  updated_at: string
  sections: EvaluationSection[]
}

export interface EvaluationSummary {
  id: string
  model_id: string
  created_at?: string
  updated_at?: string
  s3_key: string
  last_modified: string
  section_count?: number
  starred_count?: number
}
