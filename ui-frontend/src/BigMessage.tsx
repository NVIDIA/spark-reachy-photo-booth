/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

interface BigMessageProps {
  children: React.ReactNode;
}

export default function BigMessage({ children }: BigMessageProps) {
  return <div className="text-2xl font-bold">{children}</div>;
}
