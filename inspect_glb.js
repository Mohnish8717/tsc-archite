import fs from 'fs';

const buffer = fs.readFileSync('./predictive_ui/public/models/character.glb');
const jsonLength = buffer.readUInt32LE(12);
const jsonBuffer = buffer.slice(20, 20 + jsonLength);
const json = JSON.parse(jsonBuffer.toString('utf8'));

console.log("Nodes:");
json.nodes.forEach((n, i) => {
  if (n.name) console.log(`${i}: ${n.name}`);
});
