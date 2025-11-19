'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, MeshDistortMaterial, Float, Stars } from '@react-three/drei';
import { Suspense, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// OPTION 1: 3D Audio Waveform Sphere (Sound Reactive Visual)
export function AudioWaveSphere3D() {
  function WaveformSphere() {
    const meshRef = useRef<THREE.Mesh>(null);
    
    useFrame((state) => {
      if (meshRef.current) {
        meshRef.current.rotation.y = state.clock.elapsedTime * 0.3;
        // Simulate audio reactivity
        const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
        meshRef.current.scale.set(scale, scale, scale);
      }
    });

    return (
      <mesh ref={meshRef}>
        <sphereGeometry args={[2, 64, 64]} />
        <MeshDistortMaterial
          color="#8b5cf6"
          attach="material"
          distort={0.4}
          speed={2}
          roughness={0}
          metalness={0.8}
        />
      </mesh>
    );
  }

  return (
    <Canvas camera={{ position: [0, 0, 6], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#a855f7" />
        <pointLight position={[-10, -10, -5]} intensity={0.5} color="#ec4899" />
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        
        <WaveformSphere />
        
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={1} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 2: Circular Audio Visualizer Rings
export function AudioRings3D() {
  function RotatingSphere({ position }: { position: [number, number, number] }) {
    const meshRef = useRef<THREE.Mesh>(null);
    
    useFrame((state) => {
      if (meshRef.current) {
        meshRef.current.rotation.x = state.clock.elapsedTime * 0.5;
        meshRef.current.rotation.y = state.clock.elapsedTime * 0.3;
        meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime) * 0.5;
      }
    });

    return (
      <mesh ref={meshRef} position={position}>
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshStandardMaterial 
          color="#8b5cf6" 
          emissive="#8b5cf6"
          emissiveIntensity={0.5}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>
    );
  }

  return (
    <Canvas camera={{ position: [0, 0, 8], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#a855f7" />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#ec4899" />
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        
        <RotatingSphere position={[-2, 0, 0]} />
        <RotatingSphere position={[2, 0, 0]} />
        <RotatingSphere position={[0, 2, 0]} />
        <RotatingSphere position={[0, -2, 0]} />
        
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={2} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 2: Morphing Blob with Distortion
export function MorphingBlob3D() {
  return (
    <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <pointLight position={[-10, -10, -5]} intensity={0.5} color="#ec4899" />
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        
        <Float speed={2} rotationIntensity={1} floatIntensity={2}>
          <Sphere args={[1.5, 100, 200]} scale={1.5}>
            <MeshDistortMaterial
              color="#6366f1"
              attach="material"
              distort={0.6}
              speed={3}
              roughness={0}
              metalness={0.8}
            />
          </Sphere>
        </Float>
        
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={1} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 3: Geometric Torus Knot
export function GeometricTorusKnot3D() {
  return (
    <Canvas camera={{ position: [0, 0, 6], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} color="#a855f7" />
        <pointLight position={[-10, -10, -5]} intensity={0.5} color="#ec4899" />
        
        <Float speed={1.5} rotationIntensity={2} floatIntensity={1}>
          <TorusKnot args={[1, 0.3, 128, 32]}>
            <meshStandardMaterial
              color="#8b5cf6"
              emissive="#8b5cf6"
              emissiveIntensity={0.3}
              metalness={0.9}
              roughness={0.1}
            />
          </TorusKnot>
        </Float>
        
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={3} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 4: Wobbling Icosahedron
export function WobblingIcosahedron3D() {
  return (
    <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.5} />
        <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} color="#a855f7" />
        <pointLight position={[-10, -10, -5]} intensity={0.5} color="#ec4899" />
        
        <Float speed={2} rotationIntensity={1.5} floatIntensity={2}>
          <Icosahedron args={[2, 1]}>
            <MeshWobbleMaterial
              color="#6366f1"
              attach="material"
              factor={0.6}
              speed={2}
              roughness={0}
              metalness={0.8}
            />
          </Icosahedron>
        </Float>
        
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={2} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 5: Rotating Ring System
export function RotatingRings3D() {
  function AnimatedRing({ radius, speed, color }: { radius: number; speed: number; color: string }) {
    const meshRef = useRef<THREE.Mesh>(null);
    
    useFrame((state) => {
      if (meshRef.current) {
        meshRef.current.rotation.x = state.clock.elapsedTime * speed;
        meshRef.current.rotation.y = state.clock.elapsedTime * speed * 0.5;
      }
    });

    return (
      <mesh ref={meshRef}>
        <torusGeometry args={[radius, 0.1, 16, 100]} />
        <meshStandardMaterial 
          color={color}
          emissive={color}
          emissiveIntensity={0.5}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>
    );
  }

  return (
    <Canvas camera={{ position: [0, 0, 8], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        
        <AnimatedRing radius={2} speed={1} color="#8b5cf6" />
        <AnimatedRing radius={2.5} speed={-0.8} color="#ec4899" />
        <AnimatedRing radius={3} speed={0.6} color="#6366f1" />
        
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} enablePan={false} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 6: DNA Helix Structure
export function DNAHelix3D() {
  function Helix() {
    const groupRef = useRef<THREE.Group>(null);
    
    useFrame((state) => {
      if (groupRef.current) {
        groupRef.current.rotation.y = state.clock.elapsedTime * 0.5;
      }
    });

    const spheres = [];
    for (let i = 0; i < 20; i++) {
      const t = (i / 20) * Math.PI * 4;
      const x = Math.cos(t) * 1.5;
      const y = (i - 10) * 0.3;
      const z = Math.sin(t) * 1.5;
      
      spheres.push(
        <mesh key={i} position={[x, y, z]}>
          <sphereGeometry args={[0.15, 16, 16]} />
          <meshStandardMaterial 
            color={i % 2 === 0 ? "#8b5cf6" : "#ec4899"}
            emissive={i % 2 === 0 ? "#8b5cf6" : "#ec4899"}
            emissiveIntensity={0.5}
          />
        </mesh>
      );
    }

    return <group ref={groupRef}>{spheres}</group>;
  }

  return (
    <Canvas camera={{ position: [0, 0, 8], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#a855f7" />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#ec4899" />
        
        <Helix />
        
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} enablePan={false} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 7: Pulsing Cubes Grid
export function PulsingCubes3D() {
  function AnimatedCube({ position, delay }: { position: [number, number, number]; delay: number }) {
    const meshRef = useRef<THREE.Mesh>(null);
    
    useFrame((state) => {
      if (meshRef.current) {
        const scale = 1 + Math.sin(state.clock.elapsedTime * 2 + delay) * 0.3;
        meshRef.current.scale.set(scale, scale, scale);
        meshRef.current.rotation.x = state.clock.elapsedTime * 0.5;
        meshRef.current.rotation.y = state.clock.elapsedTime * 0.3;
      }
    });

    return (
      <mesh ref={meshRef} position={position}>
        <boxGeometry args={[0.5, 0.5, 0.5]} />
        <meshStandardMaterial 
          color="#8b5cf6"
          emissive="#8b5cf6"
          emissiveIntensity={0.3}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>
    );
  }

  const cubes = [];
  for (let x = -2; x <= 2; x++) {
    for (let y = -2; y <= 2; y++) {
      cubes.push(
        <AnimatedCube 
          key={`${x}-${y}`} 
          position={[x * 1.5, y * 1.5, 0]} 
          delay={x + y} 
        />
      );
    }
  }

  return (
    <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#a855f7" />
        
        {cubes}
        
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={1} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 8: Crystalline Structure
export function CrystallineStructure3D() {
  return (
    <Canvas camera={{ position: [0, 0, 6], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.4} />
        <spotLight position={[10, 10, 10]} angle={0.3} penumbra={1} intensity={2} color="#a855f7" />
        <spotLight position={[-10, -10, -10]} angle={0.3} penumbra={1} intensity={1} color="#ec4899" />
        
        <Float speed={1} rotationIntensity={2} floatIntensity={1}>
          <group>
            <Icosahedron args={[1.5, 0]} position={[0, 0, 0]}>
              <meshPhysicalMaterial
                color="#6366f1"
                transmission={0.9}
                opacity={0.8}
                metalness={0.2}
                roughness={0}
                ior={2.5}
                thickness={1}
                transparent
              />
            </Icosahedron>
            
            <Sphere args={[0.8, 32, 32]} position={[0, 0, 0]}>
              <meshStandardMaterial
                color="#ec4899"
                emissive="#ec4899"
                emissiveIntensity={0.5}
                metalness={0.9}
                roughness={0.1}
              />
            </Sphere>
          </group>
        </Float>
        
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={2} />
      </Suspense>
    </Canvas>
  );
}
