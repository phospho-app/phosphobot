'use client';

import SubscribeButton from './subscribe-button';

interface PricingCardProps {
    userEmail: string | null;
}

export default function PricingCard({ userEmail }: PricingCardProps) {
    return (
        <div className="bg-white rounded-lg shadow-[0_10px_15px_rgba(0,0,0,0.1)] p-6 sm:p-8 max-w-sm mx-auto lg:mx-0 lg:ml-8">
            <h2 className="text-xl sm:text-2xl font-bold font-headline text-gray-900 mb-2">
                Get phospho pro now.
            </h2>
            <p className="text-sm sm:text-base text-gray-600 mb-6">
                Turbocharge your robotics journey. Cancel anytime.
            </p>
            <div className="mb-6">
                <div className="flex items-baseline">
                    <span className="text-2xl sm:text-3xl font-bold text-gray-900">€35</span>
                    <span className="text-gray-600 ml-1">/ month</span>
                </div>
            </div>
            <SubscribeButton userEmail={userEmail} />
        </div>
    );
}