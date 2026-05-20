import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import * as THREE from 'three';
import fs from 'fs';

// To load GLTF in node, we need to mock some DOM elements or just read the file buffer and parse JSON
// Instead, a simpler way is to read the GLB file, find the JSON chunk, and extract animation names.

const buffer = fs.readFileSync(process.argv[2]);
const magic = buffer.readUInt32LE(0);
if (magic !== 0x46546C67) {
  console.log("Not a GLB");
  process.exit(1);
}

const jsonChunkLength = buffer.readUInt32LE(12);
const jsonChunkType = buffer.readUInt32LE(16);
if (jsonChunkType !== 0x4E4F534A) {
  console.log("First chunk is not JSON");
  process.exit(1);
}

const jsonStr = buffer.toString('utf8', 20, 20 + jsonChunkLength);
const json = JSON.parse(jsonStr);

console.log("Animations:");
if (json.animations) {
  json.animations.forEach(anim => console.log("- " + anim.name));
} else {
  console.log("None");
}
