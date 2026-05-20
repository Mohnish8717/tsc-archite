const fs = require('fs');
const { GLTFLoader } = require('three/examples/jsm/loaders/GLTFLoader');
const { JSDOM } = require('jsdom');
const THREE = require('three');

// Mock browser environment for Three.js loaders
const dom = new JSDOM();
global.self = global;
global.window = dom.window;
global.document = dom.window.document;
global.navigator = dom.window.navigator;
global.Blob = dom.window.Blob;
global.URL = dom.window.URL;

const loader = new GLTFLoader();
const buffer = fs.readFileSync(process.argv[2]);
const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);

loader.parse(arrayBuffer, '', (gltf) => {
  gltf.scene.traverse((child) => {
    if (child.isMesh) {
      console.log(child.name);
    }
  });
  process.exit(0);
}, (err) => {
  console.error(err);
  process.exit(1);
});
