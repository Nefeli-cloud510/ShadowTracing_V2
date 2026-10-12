export interface ExperimentValidationSnapshot {
  experimentMode: 'baseline' | 'comparative'
  controlVariables: string[]
  treatmentVariables: string[]
  hasFeatureDifference: boolean
}

function normalizeVariables(values: unknown): string[] {
  if (!Array.isArray(values)) {
    return []
  }
  const seen = new Set<string>()
  const normalized: string[] = []
  for (const value of values) {
    const item = String(value ?? '').trim()
    if (!item || seen.has(item)) {
      continue
    }
    seen.add(item)
    normalized.push(item)
  }
  return normalized
}

export function getExperimentValidationSnapshot(candidate: any): ExperimentValidationSnapshot {
  const design = candidate?.design ?? {}
  const controlVariables = normalizeVariables(
    Array.isArray(design.display_control) && design.display_control.length > 0
      ? design.display_control
      : design.control,
  )
  const treatmentVariables = normalizeVariables(
    Array.isArray(design.display_treatment) && design.display_treatment.length > 0
      ? design.display_treatment
      : design.treatment,
  )
  const experimentMode = candidate?.experiment_type === 'baseline_benchmark'
    || candidate?.type === 'baseline_benchmark'
    ? 'baseline'
    : 'comparative'
  const hasFeatureDifference = experimentMode === 'baseline'
    || JSON.stringify([...controlVariables].sort()) !== JSON.stringify([...treatmentVariables].sort())
  return {
    experimentMode,
    controlVariables,
    treatmentVariables,
    hasFeatureDifference,
  }
}

export function isSelectableCandidateExperiment(candidate: any): boolean {
  const snapshot = getExperimentValidationSnapshot(candidate)
  return snapshot.experimentMode === 'comparative' && snapshot.hasFeatureDifference
}

export function filterSelectableCandidateExperiments<T>(candidates: T[]): T[] {
  return candidates.filter((candidate) => isSelectableCandidateExperiment(candidate))
}
