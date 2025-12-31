"use client";

import React from "react";
import { motion } from "framer-motion";

export function Header() {
  return (
    <header className="text-center py-8">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="inline-flex items-center gap-3"
      >
        {/* Logo */}
        <motion.div
          className="text-5xl"
          animate={{ scale: [1, 1.1, 1], rotate: [0, 5, -5, 0] }}
          transition={{ duration: 3, repeat: Infinity, repeatDelay: 2 }}
        >
          🔮
        </motion.div>

        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
            Prism
          </h1>
          <p className="text-muted-foreground">
            Transform academic papers into adorable manga
          </p>
        </div>
      </motion.div>
    </header>
  );
}
