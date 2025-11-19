'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, MeshDistortMaterial, Float, Stars } from '@react-three/drei';
import { Suspense, useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// OPTION 1: Pulsating Sound Wave Sphere
export function SoundWaveSphere3D() {
  function PulsatingSphere() {
    const meshRef = useRef<THREE.Mesh>(null);
    
    useFrame((state) => {
      if (meshRef.current) {
        meshRef.current.rotation.y = state.clock.elapsedTime * 0.3;
        // Simulate audio pulse
        const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.15;
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
        
        <PulsatingSphere />
        
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={1} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 2: 3D Circular Audio Rings (Equalizer Style)
export function CircularAudioRings3D() {
  function AudioRing({ radius, speed, offset }: { radius: number; speed: number; offset: number }) {
    const meshRef = useRef<THREE.Mesh>(null);
    
    useFrame((state) => {
      if (meshRef.current) {
        meshRef.current.rotation.z = state.clock.elapsedTime * speed;
        // Pulsate thickness based on "audio"
        const scale = 1 + Math.sin(state.clock.elapsedTime * 2 + offset) * 0.2;
        meshRef.current.scale.setY(scale);
      }
    });

    return (
      <mesh ref={meshRef}>
        <torusGeometry args={[radius, 0.15, 16, 100]} />
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
        
        <AudioRing radius={1.5} speed={1} offset={0} />
        <AudioRing radius={2} speed={-0.8} offset={1} />
        <AudioRing radius={2.5} speed={0.6} offset={2} />
        <AudioRing radius={3} speed={-0.4} offset={3} />
        
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} enablePan={false} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 3: 3D Frequency Bars (3D Equalizer)
export function FrequencyBars3D() {
  function Bar({ position, delay }: { position: [number, number, number]; delay: number }) {
    const meshRef = useRef<THREE.Mesh>(null);
    
    useFrame((state) => {
      if (meshRef.current) {
        // Animate height like frequency bars
        const height = 1 + Math.abs(Math.sin(state.clock.elapsedTime * 2 + delay)) * 2;
        meshRef.current.scale.setY(height);
        meshRef.current.position.y = height / 2;
      }
    });

    return (
      <mesh ref={meshRef} position={position}>
        <boxGeometry args={[0.4, 1, 0.4]} />
        <meshStandardMaterial 
          color="#8b5cf6"
          emissive="#8b5cf6"
          emissiveIntensity={0.4}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>
    );
  }

  const bars = useMemo(() => {
    const barArray = [];
    const count = 12;
    const radius = 3;
    
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      barArray.push(
        <Bar key={i} position={[x, 0, z]} delay={i * 0.3} />
      );
    }
    return barArray;
  }, []);

  return (
    <Canvas camera={{ position: [0, 5, 8], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.3} />
        <pointLight position={[0, 10, 0]} intensity={1} color="#a855f7" />
        <pointLight position={[0, -5, 5]} intensity={0.5} color="#ec4899" />
        
        {bars}
        
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={1} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 4: Sound Wave Particles (Like vocal waves)
export function SoundWaveParticles3D() {
  function WaveParticles() {
    const pointsRef = useRef<THREE.Points>(null);
    
    const particles = useMemo(() => {
      const count = 1000;
      const positions = new Float32Array(count * 3);
      
      for (let i = 0; i < count; i++) {
        const i3 = i * 3;
        const radius = 2 + Math.random() * 2;
        const angle = Math.random() * Math.PI * 2;
        const height = (Math.random() - 0.5) * 5;
        
        positions[i3] = Math.cos(angle) * radius;
        positions[i3 + 1] = height;
        positions[i3 + 2] = Math.sin(angle) * radius;
      }
      
      return positions;
    }, []);
    
    useFrame((state) => {
      if (pointsRef.current) {
        pointsRef.current.rotation.y = state.clock.elapsedTime * 0.2;
        
        const positions = pointsRef.current.geometry.attributes.position.array as Float32Array;
        
        for (let i = 0; i < positions.length; i += 3) {
          const i3 = i;
          const angle = Math.atan2(positions[i3 + 2], positions[i3]);
          const radius = Math.sqrt(positions[i3] ** 2 + positions[i3 + 2] ** 2);
          
          // Create wave effect
          const wave = Math.sin(state.clock.elapsedTime * 2 + angle * 3) * 0.3;
          const newRadius = radius + wave;
          
          positions[i3] = Math.cos(angle) * newRadius;
          positions[i3 + 2] = Math.sin(angle) * newRadius;
        }
        
        pointsRef.current.geometry.attributes.position.needsUpdate = true;
      }
    });

    return (
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={particles.length / 3}
            array={particles}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial size={0.05} color="#8b5cf6" sizeAttenuation transparent opacity={0.8} />
      </points>
    );
  }

  return (
    <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#a855f7" />
        
        <WaveParticles />
        
        <Stars radius={100} depth={50} count={2000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} enablePan={false} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 5: Vocal Waveform Sphere (Microphone Visual)
export function VocalWaveformSphere3D() {
  function AnimatedMicVisual() {
    const groupRef = useRef<THREE.Group>(null);
    
    useFrame((state) => {
      if (groupRef.current) {
        groupRef.current.rotation.y = state.clock.elapsedTime * 0.5;
      }
    });

    return (
      <group ref={groupRef}>
        {/* Center sphere (microphone core) */}
        <Float speed={2} rotationIntensity={0} floatIntensity={0.5}>
          <Sphere args={[1, 32, 32]}>
            <meshStandardMaterial 
              color="#ec4899"
              emissive="#ec4899"
              emissiveIntensity={0.5}
              metalness={0.8}
              roughness={0.2}
            />
          </Sphere>
        </Float>
        
        {/* Pulsating outer rings (sound waves) */}
        {[1.5, 2, 2.5, 3].map((radius, i) => (
          <PulsingRing key={i} radius={radius} delay={i * 0.5} />
        ))}
      </group>
    );
  }
  
  function PulsingRing({ radius, delay }: { radius: number; delay: number }) {
    const meshRef = useRef<THREE.Mesh>(null);
    
    useFrame((state) => {
      if (meshRef.current) {
        const scale = 1 + Math.sin(state.clock.elapsedTime * 2 + delay) * 0.1;
        meshRef.current.scale.set(scale, scale, scale);
        
        const opacity = 0.3 - (radius / 10);
        (meshRef.current.material as THREE.MeshStandardMaterial).opacity = opacity;
      }
    });

    return (
      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 32, 32]} />
        <meshStandardMaterial 
          color="#8b5cf6"
          transparent
          opacity={0.3}
          metalness={0.5}
          roughness={0.5}
        />
      </mesh>
    );
  }

  return (
    <Canvas camera={{ position: [0, 0, 8], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#a855f7" />
        <pointLight position={[-10, -10, -5]} intensity={0.5} color="#ec4899" />
        
        <AnimatedMicVisual />
        
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} enablePan={false} />
      </Suspense>
    </Canvas>
  );
}

// OPTION 6: Radiating Sound Waves (Sonar Style)
export function RadiatingSoundWaves3D() {
  function SonarWave({ delay }: { delay: number }) {
    const meshRef = useRef<THREE.Mesh>(null);
    
    useFrame((state) => {
      if (meshRef.current) {
        const scale = ((state.clock.elapsedTime + delay) % 4) / 4 * 5;
        meshRef.current.scale.set(scale, scale, scale);
        
        const opacity = 1 - ((state.clock.elapsedTime + delay) % 4) / 4;
        (meshRef.current.material as THREE.MeshStandardMaterial).opacity = opacity * 0.5;
      }
    });

    return (
      <mesh ref={meshRef}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial 
          color="#8b5cf6"
          transparent
          opacity={0.5}
          metalness={0.5}
          roughness={0.5}
          wireframe
        />
      </mesh>
    );
  }

  return (
    <Canvas camera={{ position: [0, 0, 8], fov: 75 }}>
      <Suspense fallback={null}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#a855f7" />
        
        {/* Central emitter */}
        <Sphere args={[0.5, 32, 32]}>
          <meshStandardMaterial 
            color="#ec4899"
            emissive="#ec4899"
            emissiveIntensity={1}
          />
        </Sphere>
        
        {/* Multiple expanding waves */}
        <SonarWave delay={0} />
        <SonarWave delay={1} />
        <SonarWave delay={2} />
        <SonarWave delay={3} />
        
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableZoom={false} enablePan={false} />
      </Suspense>
    </Canvas>
  );
}
