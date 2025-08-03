"use client";

import { Button } from "@/components/ui/button";

export default function FinalCTASection() {
    return (
        <section className="py-16 px-8 bg-gray-background">
            <div className="max-w-md mx-auto bg-white rounded-xl p-8 shadow-card">
                <h2 className="text-3xl font-headline font-bold text-dark-gray mb-4">
                    Get phospho pro now.
                </h2>

                <p className="text-medium-gray mb-8">
                    Turbocharge your robotics journey. Cancel anytime.
                </p>

                <div className="flex items-baseline gap-1 mb-8">
                    <span className="text-4xl font-bold text-dark-gray">€35</span>
                    <span className="text-medium-gray">/ month</span>
                </div>

                <Button
                    className="w-full bg-primary-green hover:bg-green-500 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
                    onClick={() => window.open('https://billing.phospho.ai/b/5kQbJ2grJcfv4b369Y33W0g', '_blank')}
                >
                    subscribe →
                </Button>
            </div>
        </section>
    );
}