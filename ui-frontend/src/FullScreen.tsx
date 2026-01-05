/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

interface FullScreenProps {
  children: React.ReactNode;
}

export default function FullScreen({ children }: FullScreenProps) {
  return (
    <div className="h-screen w-screen bg-[#1a1a1a] text-white/90 relative flex items-center justify-center">
      {children}
    </div>
  );
}
