/**
 * Types for Counter-Defense / Active Defense module.
 * Backend: attacker_profiles, counter_defense_actions (when implemented).
 */

export interface AttackerProfile {
  id: string
  ip_addresses: string[]
  geolocation?: {
    country?: string
    country_code?: string
    city?: string
    latitude?: number
    longitude?: number
  }
  asn_info?: { asn?: string; org?: string }
  risk_score: number
  first_seen: string
  last_seen: string
  attack_count: number
  threat_actor_id?: string
  counter_measures_applied?: string[]
}

export interface CounterDefenseAction {
  id: string
  attacker_profile_id: string
  action_type: string
  action_params?: Record<string, unknown>
  executed_by: string
  execution_result?: Record<string, unknown>
  timestamp: string
  legal_approval?: boolean
}

export type CounterMeasureType =
  | 'block_ip'
  | 'honeypot_redirect'
  | 'tarpit'
  | 'lock_accounts'
  | 'gather_intel'
  | 'deploy_deception'
  | 'execute_playbook'
