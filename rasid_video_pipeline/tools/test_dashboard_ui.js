// Exercise the page's actual JavaScript with a minimal DOM, without a browser window.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
function element() {
  const classes = new Set();
  return {textContent:'', value:'red top', disabled:false,
    classList:{add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c)},
    children:[], replaceChildren(){this.children=[];}, appendChild(node){this.children.push(node);}, style:{}};
}
const document = {getElementById(id) {if(!elements.has(id)) elements.set(id,element());return elements.get(id);},createElement:element};
class WebSocket {static OPEN=1;constructor(){this.readyState=1;}send(payload){this.last=JSON.parse(payload);}}
const ctx=vm.createContext({document,WebSocket,console,setTimeout(){},alert(){}});
vm.runInContext(script,ctx);
const call=s=>vm.runInContext(s,ctx);
// Idle/crowd-only activity must not imply an active missing-person search.
assert(html.includes('id="searchStatusTitle">\n                        Ready to search'));
for (const [state, title, busy] of [
  ['idle','Ready to search',false],
  ['searching','Searching for matching candidates',true],
  ['awaiting_decision','Candidate found — review required',false],
  ['confirming','Applying confirmation',true],
  ['resuming_search','Applying rejection',true],
  ['confirmed','Confirmed — tracking person',true],
  ['finished','Search stopped',false],
]) {
  call(`handleBackendMessage(${JSON.stringify({type:'search_status',state,message:'Backend message'})})`);
  assert.strictEqual(document.getElementById('searchStatusTitle').textContent,title);
  assert.strictEqual(document.getElementById('searchActivityIndicator').hidden,!busy);
}
call('handleBackendMessage({type:"search_status",state:"idle",message:"Ready to search."})');
call('handleBackendMessage({type:"crowd_snapshot",sim_time:0,image:"data:image/jpeg;base64,example"})');
assert.strictEqual(document.getElementById('searchStatusTitle').textContent,'Ready to search');
assert.strictEqual(document.getElementById('searchActivityIndicator').hidden,true);
call('startSearch()');
assert.strictEqual(document.getElementById('searchStatusTitle').textContent,'Starting search');
call('handleBackendMessage({type:"search_status",state:"idle",message:"Ready to search."})');
console.log('Search state headings and activity indicators passed.');
call('handleBackendMessage({type:"candidate",candidate_id:"one",image:"data:image/jpeg;base64,example",justification:"Red top"})');
call('confirmCandidate()');
assert(!document.getElementById('confirmedCard').classList.contains('visible'));
assert(document.getElementById('candidateCard').classList.contains('visible'));
call('handleBackendMessage({type:"search_status",state:"confirming",message:"Pending"})');
assert(!document.getElementById('confirmedCard').classList.contains('visible'));
call('handleBackendMessage({type:"candidate",candidate_id:"one",image:"data:image/jpeg;base64,example",justification:"Reconnected candidate"})');
assert(document.getElementById('confirmCandidateButton').disabled);
call('handleBackendMessage({type:"search_status",state:"confirmed",message:"Confirmed"})');
assert(document.getElementById('confirmedCard').classList.contains('visible'));
call('handleBackendMessage({type:"person_location",location:{lat:24,lon:46},drone:{lat:25,lon:47},position_sim_time:12,sim_time:29})');
assert(document.getElementById('personLocation').textContent.includes('Person: 24, 46'));
assert(document.getElementById('personLocation').textContent.includes('Drone: 25, 47'));
assert(document.getElementById('personLocation').textContent.includes('seen 17 s ago'));
call('handleBackendMessage({type:"person_location",location:{lat:24,lon:46},drone:null,position_sim_time:29,sim_time:29})');
assert(document.getElementById('personLocation').textContent.includes('seen 0 s ago'));
call('handleBackendMessage({type:"person_location",location:null,drone:null,position_sim_time:null,sim_time:null})');
assert(document.getElementById('personLocation').textContent.includes('observation age unavailable'));

call('handleBackendMessage({type:"lost"})');
assert(!document.getElementById('confirmedCard').classList.contains('visible'));
call('handleBackendMessage({type:"gpt_verification_off",message:"GPT verification is OFF"})');
call('handleBackendMessage({type:"search_status",state:"searching",message:"Searching",gpt_verification_off:true})');
assert(document.getElementById('gptWarning').textContent.includes('OFF'));
call('handleBackendMessage({type:"search_status",state:"searching",message:"Fresh search",gpt_verification_off:false})');
assert.strictEqual(document.getElementById('gptWarning').textContent,'');
console.log('Dashboard UI acknowledgement, location, loss and persistent warning tests passed.');

call('handleBackendMessage({type:"crowd_analytics",sim_time:100,total_people_in_view:19,zones:{walkway:{display_name:"Custom walkway label",count:5,capacity:120,coverage:"partial",coverage_fraction:0.5,occupancy_percent:null,crowded:null,last_seen_sim_time:100},plaza:{count:1,capacity:250,coverage:"full",coverage_fraction:0.9,occupancy_percent:0.4,crowded:false,last_seen_sim_time:100},entrance:{count:null,capacity:60,coverage:"unseen",coverage_fraction:0,occupancy_percent:null,crowded:null,last_seen_sim_time:83}}})');
assert.strictEqual(document.getElementById('totalPeople').textContent,19);
assert(document.getElementById('zonesContainer').children[0].innerHTML.includes('Custom walkway label'));
assert.strictEqual(document.getElementById('totalFootfall').textContent,'—');
assert.strictEqual(call('getOccupancy({coverage:"full",coverage_fraction:0.9,occupancy_percent:0.4})'),0.4);
assert.strictEqual(call('getOccupancy({coverage:"partial",coverage_fraction:0.5,occupancy_percent:null})'),null);
assert(document.getElementById('zonesContainer').children[0].innerHTML.includes('partly visible'));
assert(document.getElementById('zonesContainer').children[0].innerHTML.includes('people in view'));
assert(!document.getElementById('zonesContainer').children[0].innerHTML.includes('dwell'));
assert(document.getElementById('zonesContainer').children[1].innerHTML.includes('0.4%'));
assert.strictEqual(document.getElementById('zoneAge-entrance').textContent,'seen 17 s ago');
call('handleBackendMessage({type:"crowd_snapshot",sim_time:100,image:"data:image/jpeg;base64,example"})');
assert.strictEqual(document.getElementById('crowdSnapshotImage').hidden,false);
assert(document.getElementById('crowdSnapshotImage').src.startsWith('data:image/jpeg;base64,'));
call('handleBackendMessage({type:"search_status",sim_time:105,state:"confirmed",message:"tracking"})');
assert.strictEqual(document.getElementById('zoneAge-entrance').textContent,'seen 22 s ago');
console.log('Crowd snapshot, partial coverage, unknown occupancy, exact percentages and zone ages passed.');

// Reconnecting to a restarted idle backend clears the previous confirmed target.
call('handleBackendMessage({type:"candidate",candidate_id:"old",image:"data:image/jpeg;base64,example",justification:"test"})');
call('handleBackendMessage({type:"search_status",state:"confirmed",message:"Confirmed"})');
call('handleBackendMessage({type:"person_location",location:{lat:24,lon:46},drone:null,position_sim_time:100,sim_time:100})');
call('handleBackendMessage({type:"search_status",state:"idle",message:"Ready to search."})');
assert(!document.getElementById('confirmedCard').classList.contains('visible'));
assert(!document.getElementById('candidateCard').classList.contains('visible'));
assert.strictEqual(document.getElementById('personLocation').textContent,'Location unavailable');
assert.strictEqual(call('currentCandidateId'),null);
assert.strictEqual(call('confirmedAcknowledged'),false);
assert.strictEqual(document.getElementById('startSearchButton').disabled,false);
call('startSearch()');
call('handleBackendMessage({type:"error",code:"command_rejected",message:"Search refused."})');
// The backend follows a rejected start with the authoritative search_status.
call('handleBackendMessage({type:"search_status",state:"idle",message:"Ready to search."})');
assert.strictEqual(document.getElementById('searchStatusTitle').textContent,'Ready to search');
assert.strictEqual(document.getElementById('searchActivityIndicator').hidden,true);
assert.strictEqual(document.getElementById('searchError').textContent,'Search refused.');
console.log('Idle reconnect clears stale cards; rejected-start state restoration passed.');
