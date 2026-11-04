import { actionName, cardName, type GameName, type Info } from '../lib/api';

export function ErrorBox({error,retry}:{error:string;retry?:()=>void}) {
  return <div className="error" role="alert" aria-label="Request error"><span>{error}</span>{retry && <button className="text-button" onClick={retry}>Try again</button>}</div>;
}
export function Loading({label='Loading recorded data…'}:{label?:string}) {
  return <div className="loading" role="status"><span className="spinner"/>{label}</div>;
}
export function Card({game,card,hidden=false,small=false}:{game:GameName;card?:number|null;hidden?:boolean;small?:boolean}) {
  const name=cardName(game,card);
  const red=card!=null && (game==='leduc'?card%2===1:game==='holdem'&&(card%4===1||card%4===2));
  return <div className={`playing-card ${hidden || card==null ? 'card-back':''} ${red?'red':''} ${small?'small-card':''}`} aria-label={hidden?'Hidden AI card':name}>
    {hidden || card==null ? <span className="back-mark">♧</span> : <><span className="card-corner">{name}</span><span className="card-center">{game==='kuhn'?name:name.slice(-1)}</span><span className="card-bottom">{name}</span></>}
  </div>;
}
export function Probabilities({info,probabilities,chosen}:{info:Info;probabilities:number[];chosen?:number}) {
  return <div className="probabilities">{info.legal_actions.map(action=><div key={action}>
    <div className="prob-label"><span>{actionName(info,action)} {chosen===action && <span className="chosen">played</span>}</span><strong>{(probabilities[action]*100).toFixed(1)}%</strong></div>
    <div className="prob-track"><div style={{width:`${probabilities[action]*100}%`}}/></div>
  </div>)}</div>;
}
