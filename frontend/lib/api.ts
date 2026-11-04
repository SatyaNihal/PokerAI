export type GameName = 'kuhn' | 'leduc' | 'holdem';
export type Agent = {id:string; game:GameName; algorithm:string; name:string; seed:number; experiment_id:string; metrics:Record<string, number>; inference_mode:string};
export type PublicAction = {player:number; round:number; action:number; label:string};
export type Game = {id:string; game:GameName; agent_id:string; human_seat:number; version:number; status:'playing'|'complete'; current_player:number|null; private_card:number; public_card:number|null; private_cards?:number[]; public_cards?:number[]; round:number; pot:number; contributions:number[]; stacks:number[]|null; history:PublicAction[]; legal_actions:{id:number;label:string;cost:number}[]; returns:number[]|null};
export type Info = {key:string; game:GameName; player:number; private_card:number; public_card:number|null; private_cards?:number[]; public_cards?:number[]; history:number[][]; legal_actions:number[]};
export type Policy = {agent_id:string; experiment_id:string; encoder_version:string; information_state:Info; tensor:number[]; probabilities:number[]};
export type Analysis = {game_id:string; ai_card:number; ai_cards?:number[]; ai_seat:number; decisions:{information_state:Info; probabilities:number[]; action:number}[]};
export type BoardRunout = {game_id:string; public_cards:number[]};
export type Metric = {iteration:number; exploitability:number; nash_conv:number; elapsed_seconds:number; training_seconds:number; evaluation_seconds:number; nodes_touched:number; samples_processed:number; advantage_loss?:number|null; strategy_loss?:number|null; advantage_samples_generated?:number; strategy_samples_generated?:number; [key:string]:unknown};
export type Experiment = {experiment_id:string; status:string; config:{game:GameName;algorithm:string;seed:number;[key:string]:unknown}; final_metrics:Metric; wall_clock_seconds:number; metrics:Metric[]; provenance:Record<string,unknown>};
export type RunSummary = {experiment_id:string;game:GameName;algorithm:string;seed:number;iterations:number;exploitability:number;nash_conv:number;value_player0:number;offline_wall_seconds:number;peak_sampled_rss_bytes:number;samples_processed:number;nodes_touched:number};
export type Tournament = {game:GameName;agent_a:string;agent_b:string;payoff_per_hand:number;ci_low:number;ci_high:number;exact_payoff_per_hand:number;pairs:number;hands:number};
export type Summary = {registration:{selected_profile:string;[key:string]:unknown};runs:RunSummary[];summaries:{game:GameName;algorithm:string;run_count:number;mean_exploitability:number;seed_ci95:number[]|null;deterministic:boolean;mean_offline_seconds:number;experiment_ids:string[]}[]; tournaments:Tournament[];benchmark_experiment:string;tournament_experiment:string};

export async function api<T>(path:string, body?:unknown):Promise<T> {
  const response = await fetch((process.env.NEXT_PUBLIC_API_BASE_URL || '') + path, body === undefined ? {cache:'no-store'} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await response.json().catch(()=>({detail:'The server returned an unreadable response.'}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be completed.');
  return data as T;
}

export const methodName = (value:string) => ({cfr:'CFR', cfr_plus:'CFR+', deep_cfr:'Deep CFR', uniform:'Uniform', rule_based:'Rule based', always_call:'Always call'}[value] || value.replace('deep_cfr_s','Deep CFR · seed '));
export const number = (value:number, digits=3) => value !== 0 && (Math.abs(value)<Math.max(0.001,10**(-digits)) || Math.abs(value)>=1e7) ? value.toExponential(2) : value.toLocaleString('en-US',{maximumFractionDigits:digits});
export const cardName = (game:GameName, card:number|null|undefined) => card == null ? 'Undealt' : game === 'kuhn' ? ['J','Q','K'][card] : game==='holdem' ? ['2','3','4','5','6','7','8','9','10','J','Q','K','A'][Math.floor(card/4)]+'♣♦♥♠'[card%4] : ['J♠','J♥','Q♠','Q♥','K♠','K♥'][card];
export function actionName(info:Info, action:number) {
  const history=info.history.at(-1) || [];
  if (info.game==='kuhn') return action===0 ? (history.at(-1)===1?'Fold':'Check') : (history.at(-1)===1?'Call':'Bet');
  if(info.game==='holdem')return action===0?'Fold':action===2?'Raise':(history.at(-1)===2||(info.history.length===1&&history.length===0))?'Call':'Check';
  return action===0?'Fold':action===2?'Raise':history.at(-1)===2?'Call':'Check';
}

export type PairScore={payoff_per_hand:number;ci_low:number;ci_high:number;pairs:number;hands:number};
export type HoldemMetric={iteration:number;nodes_touched:number;elapsed_seconds:number;training_seconds:number;evaluation_seconds:number;peak_sampled_rss_bytes:number;samples_processed:number;advantage_loss:number;strategy_loss:number;tournaments:Record<string,PairScore>};
export type HoldemSummary={game:'holdem';method:string;evaluation:string;exact_exploitability:null;config:Record<string,unknown>;registration:{demo_seed:number;config_sha256:string};runs:{experiment_id:string;seed:number;checkpoint:string;wall_clock_seconds:number;metrics:HoldemMetric[]}[];response_probes:{experiment_id:string;seed:number;response_gap:number;ci_low:number;ci_high:number;pairs:number;exact_exploitability:false}[];limitations:string[]};
export function historyName(info:Info) {
  return info.history.map((round,index)=>`${index+1}: ${round.length ? round.map(a=>info.game==='kuhn' ? (a?'b':'p') : ['f','c','r'][a]).join(' → ') : 'start'}`).join(' / ');
}
