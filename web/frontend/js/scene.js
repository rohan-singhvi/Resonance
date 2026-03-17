import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const COLORS = {
    source: 0xf97316,
    listener: 0x06b6d4,
    floor: 0x3f3f46,
    walls: 0x27272a,
    ceiling: 0x27272a,
    grid: 0x27272a,
    ray: 0x3b82f6,
    mesh: 0x6366f1,
};

const MATERIAL_COLORS = {
    concrete: 0x6b7280, brick: 0x92400e, drywall: 0x9ca3af,
    plaster: 0xd1d5db, glass: 0x7dd3fc, wood_floor: 0x92400e,
    carpet_thin: 0x7c3aed, carpet_thick: 0x6d28d9,
    acoustic_foam: 0x0d9488, acoustic_panel: 0x0891b2,
    curtain: 0xbe185d, audience: 0x9333ea,
};

export class RoomScene {
    constructor(canvas, onDrag) {
        this.canvas = canvas;
        this.rayLines = [];
        this.onDrag = onDrag || null;
        this.roomBounds = null;
        this._init();
        this._initDrag();
        this._animate = this._animate.bind(this);
        this._animate();
        window.addEventListener('resize', () => this._resize());
    }

    _init() {
        const w = this.canvas.parentElement.clientWidth;
        const h = this.canvas.parentElement.clientHeight;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0c);
        this.scene.fog = new THREE.Fog(0x0a0a0c, 60, 120);

        this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 200);
        this.camera.position.set(14, 10, 14);

        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false,
        });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(w, h);
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;

        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 2;
        this.controls.maxDistance = 100;

        // Lights
        const ambient = new THREE.AmbientLight(0xffffff, 0.3);
        this.scene.add(ambient);

        const dir = new THREE.DirectionalLight(0xffffff, 0.6);
        dir.position.set(10, 20, 10);
        this.scene.add(dir);

        // Grid
        this.grid = new THREE.GridHelper(100, 100, COLORS.grid, COLORS.grid);
        this.grid.material.opacity = 0.15;
        this.grid.material.transparent = true;
        this.scene.add(this.grid);

        // Room group
        this.roomGroup = new THREE.Group();
        this.scene.add(this.roomGroup);

        // Source sphere
        const srcGeo = new THREE.IcosahedronGeometry(0.15, 2);
        const srcMat = new THREE.MeshStandardMaterial({
            color: COLORS.source, emissive: COLORS.source, emissiveIntensity: 0.8,
        });
        this.sourceMesh = new THREE.Mesh(srcGeo, srcMat);
        this.scene.add(this.sourceMesh);

        // Source glow
        this.sourceLight = new THREE.PointLight(COLORS.source, 2, 8);
        this.sourceMesh.add(this.sourceLight);

        // Listener sphere
        const lisGeo = new THREE.IcosahedronGeometry(0.15, 2);
        const lisMat = new THREE.MeshStandardMaterial({
            color: COLORS.listener, emissive: COLORS.listener, emissiveIntensity: 0.6,
        });
        this.listenerMesh = new THREE.Mesh(lisGeo, lisMat);
        this.scene.add(this.listenerMesh);

        // Ray group
        this.rayGroup = new THREE.Group();
        this.scene.add(this.rayGroup);

        // Initial room
        this.updateRoom('shoebox', [10, 5, 8], {});
        this.setSource([2, 1.5, 1.5]);
        this.setListener([8, 1.5, 1.5]);
    }

    _resize() {
        const w = this.canvas.parentElement.clientWidth;
        const h = this.canvas.parentElement.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    _initDrag() {
        this._raycaster = new THREE.Raycaster();
        this._mouse = new THREE.Vector2();
        this._dragTarget = null;
        this._dragPlane = new THREE.Plane();
        this._intersection = new THREE.Vector3();

        // Make spheres bigger hit targets
        const hitGeo = new THREE.SphereGeometry(0.4, 8, 8);
        const hitMat = new THREE.MeshBasicMaterial({ visible: false });

        this._sourceHit = new THREE.Mesh(hitGeo, hitMat);
        this.sourceMesh.add(this._sourceHit);

        this._listenerHit = new THREE.Mesh(hitGeo.clone(), hitMat.clone());
        this.listenerMesh.add(this._listenerHit);

        this.canvas.addEventListener('pointerdown', (e) => this._onPointerDown(e));
        this.canvas.addEventListener('pointermove', (e) => this._onPointerMove(e));
        this.canvas.addEventListener('pointerup', () => this._onPointerUp());
        this.canvas.addEventListener('pointerleave', () => this._onPointerUp());
    }

    _getNDC(e) {
        const rect = this.canvas.getBoundingClientRect();
        this._mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this._mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    }

    _clamp(x, y, z) {
        const b = this.roomBounds;
        if (!b) return [x, y, z];
        const pad = 0.2;
        return [
            Math.max(b.min.x + pad, Math.min(b.max.x - pad, x)),
            Math.max(b.min.y + pad, Math.min(b.max.y - pad, y)),
            Math.max(b.min.z + pad, Math.min(b.max.z - pad, z)),
        ];
    }

    _onPointerDown(e) {
        this._getNDC(e);
        this._raycaster.setFromCamera(this._mouse, this.camera);
        const hits = this._raycaster.intersectObjects([this._sourceHit, this._listenerHit]);
        if (hits.length === 0) return;

        const hit = hits[0].object;
        this._dragTarget = hit === this._sourceHit ? 'source' : 'listener';
        const mesh = this._dragTarget === 'source' ? this.sourceMesh : this.listenerMesh;

        // Drag plane: horizontal (y = mesh.y), facing camera
        this._dragPlane.setFromNormalAndCoplanarPoint(
            new THREE.Vector3(0, 1, 0),
            mesh.position
        );

        this.controls.enabled = false;
        this.canvas.style.cursor = 'grabbing';
    }

    _onPointerMove(e) {
        if (!this._dragTarget) {
            this._getNDC(e);
            this._raycaster.setFromCamera(this._mouse, this.camera);
            const hits = this._raycaster.intersectObjects([this._sourceHit, this._listenerHit]);
            this.canvas.style.cursor = hits.length > 0 ? 'grab' : '';
            return;
        }

        this._getNDC(e);
        this._raycaster.setFromCamera(this._mouse, this.camera);
        if (!this._raycaster.ray.intersectPlane(this._dragPlane, this._intersection)) return;

        const mesh = this._dragTarget === 'source' ? this.sourceMesh : this.listenerMesh;
        const [cx, , cz] = this._clamp(this._intersection.x, mesh.position.y, this._intersection.z);
        mesh.position.x = cx;
        mesh.position.z = cz;

        if (this.onDrag) {
            const p = mesh.position;
            this.onDrag(this._dragTarget, [
                Math.round(p.x * 10) / 10,
                Math.round(p.y * 10) / 10,
                Math.round(p.z * 10) / 10,
            ]);
        }
    }

    _onPointerUp() {
        if (!this._dragTarget) return;
        this._dragTarget = null;
        this.controls.enabled = true;
        this.canvas.style.cursor = '';
    }

    _animate() {
        requestAnimationFrame(this._animate);
        this.controls.update();

        // Pulse source glow
        const t = performance.now() * 0.002;
        this.sourceLight.intensity = 1.5 + Math.sin(t) * 0.5;

        this.renderer.render(this.scene, this.camera);
    }

    resetCamera() {
        this.controls.reset();
        this.camera.position.set(14, 10, 14);
        this.controls.target.set(0, 0, 0);
    }

    setSource(pos) {
        this.sourceMesh.position.set(pos[0], pos[1], pos[2]);
    }

    setListener(pos) {
        this.listenerMesh.position.set(pos[0], pos[1], pos[2]);
    }

    updateRoom(type, dims, materials) {
        // Clear old room
        while (this.roomGroup.children.length) {
            const c = this.roomGroup.children[0];
            c.geometry?.dispose();
            c.material?.dispose();
            this.roomGroup.remove(c);
        }

        if (type === 'shoebox') {
            this._buildShoebox(dims, materials);
            this.roomBounds = new THREE.Box3(
                new THREE.Vector3(0, 0, 0),
                new THREE.Vector3(dims[0], dims[1], dims[2])
            );
        } else if (type === 'dome') {
            this._buildDome(dims[0], materials);
            this.roomBounds = new THREE.Box3(
                new THREE.Vector3(-dims[0], 0, -dims[0]),
                new THREE.Vector3(dims[0], dims[0], dims[0])
            );
        } else if (type === 'mesh') {
            return; // mesh rooms set bounds via loadMesh
        }

        // Re-center camera target
        const cx = type === 'dome' ? 0 : dims[0] / 2;
        const cy = type === 'dome' ? dims[0] / 3 : dims[1] / 2;
        const cz = type === 'dome' ? 0 : dims[2] / 2;
        this.controls.target.set(cx, cy, cz);

        const maxDim = Math.max(...dims);
        const dist = maxDim * 1.5;
        this.camera.position.set(cx + dist * 0.7, cy + dist * 0.5, cz + dist * 0.7);
    }

    loadMesh(objText) {
        // Clear old room
        while (this.roomGroup.children.length) {
            const c = this.roomGroup.children[0];
            c.geometry?.dispose();
            c.material?.dispose();
            this.roomGroup.remove(c);
        }

        const loader = new OBJLoader();
        const obj = loader.parse(objText);

        // Wireframe + translucent fill
        obj.traverse((child) => {
            if (child.isMesh) {
                // Translucent fill
                child.material = new THREE.MeshStandardMaterial({
                    color: COLORS.mesh,
                    transparent: true,
                    opacity: 0.1,
                    side: THREE.DoubleSide,
                    roughness: 0.9,
                });

                // Wireframe overlay
                const edges = new THREE.EdgesGeometry(child.geometry);
                const edgeMat = new THREE.LineBasicMaterial({ color: 0x525252 });
                const wire = new THREE.LineSegments(edges, edgeMat);
                child.add(wire);
            }
        });

        this.roomGroup.add(obj);

        // Compute bounds
        const box = new THREE.Box3().setFromObject(obj);
        this.roomBounds = box;

        // Center camera
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        this.controls.target.copy(center);
        this.camera.position.set(
            center.x + maxDim * 0.7,
            center.y + maxDim * 0.5,
            center.z + maxDim * 0.7
        );

        return { bounds: box, center, size };
    }

    _buildShoebox(dims, materials) {
        const [x, y, z] = dims;
        const geo = new THREE.BoxGeometry(x, y, z);

        // Wireframe edges
        const edges = new THREE.EdgesGeometry(geo);
        const edgeMat = new THREE.LineBasicMaterial({ color: 0x525252, linewidth: 1 });
        const wireframe = new THREE.LineSegments(edges, edgeMat);
        wireframe.position.set(x / 2, y / 2, z / 2);
        this.roomGroup.add(wireframe);

        // Translucent faces per surface
        const faces = [
            { name: 'floor', geo: new THREE.PlaneGeometry(x, z), pos: [x/2, 0, z/2], rot: [-Math.PI/2, 0, 0] },
            { name: 'ceiling', geo: new THREE.PlaneGeometry(x, z), pos: [x/2, y, z/2], rot: [Math.PI/2, 0, 0] },
            { name: 'walls', geo: new THREE.PlaneGeometry(x, y), pos: [x/2, y/2, 0], rot: [0, 0, 0] },
            { name: 'walls', geo: new THREE.PlaneGeometry(x, y), pos: [x/2, y/2, z], rot: [0, Math.PI, 0] },
            { name: 'walls', geo: new THREE.PlaneGeometry(z, y), pos: [0, y/2, z/2], rot: [0, Math.PI/2, 0] },
            { name: 'walls', geo: new THREE.PlaneGeometry(z, y), pos: [x, y/2, z/2], rot: [0, -Math.PI/2, 0] },
        ];

        for (const f of faces) {
            const matName = materials[f.name] || 'concrete';
            const color = MATERIAL_COLORS[matName] || 0x3f3f46;
            const mat = new THREE.MeshStandardMaterial({
                color,
                transparent: true,
                opacity: 0.12,
                side: THREE.DoubleSide,
                roughness: 0.9,
            });
            const mesh = new THREE.Mesh(f.geo, mat);
            mesh.position.set(...f.pos);
            mesh.rotation.set(...f.rot);
            this.roomGroup.add(mesh);
        }
    }

    _buildDome(radius, materials) {
        // Floor
        const floorGeo = new THREE.CircleGeometry(radius, 48);
        const floorMat = new THREE.MeshStandardMaterial({
            color: MATERIAL_COLORS[materials.floor || 'concrete'] || 0x3f3f46,
            transparent: true, opacity: 0.15, side: THREE.DoubleSide,
        });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        this.roomGroup.add(floor);

        // Dome hemisphere
        const domeGeo = new THREE.SphereGeometry(radius, 32, 24, 0, Math.PI * 2, 0, Math.PI / 2);
        const domeMat = new THREE.MeshStandardMaterial({
            color: MATERIAL_COLORS[materials.walls || 'concrete'] || 0x27272a,
            transparent: true, opacity: 0.08, side: THREE.DoubleSide,
        });
        this.roomGroup.add(new THREE.Mesh(domeGeo, domeMat));

        // Wireframe
        const wireGeo = new THREE.SphereGeometry(radius, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2);
        const wireEdges = new THREE.EdgesGeometry(wireGeo);
        const wireMat = new THREE.LineBasicMaterial({ color: 0x3f3f46, transparent: true, opacity: 0.4 });
        this.roomGroup.add(new THREE.LineSegments(wireEdges, wireMat));
    }

    showRays(rayPaths) {
        this.clearRays();
        for (let r = 0; r < rayPaths.length; r++) {
            const path = rayPaths[r];
            if (path.length < 2) continue;

            const points = path.map(p => new THREE.Vector3(p[0], p[1], p[2]));
            const geo = new THREE.BufferGeometry().setFromPoints(points);

            // Fade opacity per ray
            const opacity = 0.15 + 0.25 * (1 - r / rayPaths.length);
            const mat = new THREE.LineBasicMaterial({
                color: COLORS.ray,
                transparent: true,
                opacity,
            });
            const line = new THREE.Line(geo, mat);
            this.rayGroup.add(line);
            this.rayLines.push(line);
        }
    }

    clearRays() {
        for (const line of this.rayLines) {
            line.geometry.dispose();
            line.material.dispose();
            this.rayGroup.remove(line);
        }
        this.rayLines = [];
    }
}
