"use client";

import { useState } from 'react';
import { Plus, Minus } from 'lucide-react';

interface FAQItem {
    question: string;
    answer: string;
}

const faqData: FAQItem[] = [
    {
        question: "How do I access phospho pro features?",
        answer: "Click the **Subscribe** button and enter the same email you used for your Phospho account.\n\nAfter submitting your payment details, you'll be asked to fill in a short form with your **Meta Quest username** and **Discord username**. This helps us sync your access across platforms.\n\nYou'll get full access to Pro features within 24 hours.\n\nIf you have any questions, feel free to contact us at contact@phospho.ai."
    },
    {
        question: "How can I cancel my plan?",
        answer: "To manage your subscription, simply use the Stripe link sent to your email when you signed up.\n\nNeed help? Contact us anytime at contact@phospho.ai."
    },
    {
        question: "Why does training longer matter?",
        answer: "Training time directly impacts how well your robot learns. Researchers from Stanford, Berkeley, and Meta recommend training ACT models for \"very long\" durations. In their experiments, they trained each task for over 5 hours.\n\nSimilarly, NVIDIA's Gr00t team trains their models for 10× longer than the default training time on Phosphobot.\n\nLonger training helps models generalize better and perform more reliably, but it requires powerful GPUs and expensive hardware, which phosphobot helps you access and manage efficiently."
    },
    {
        question: "What robots are compatible?",
        answer: "phosphobot is open source and designed to work with virtually any robot.\n\nYou can find the current list of supported robots on our [GitHub](https://github.com/phospho-app/phosphobot)."
    },
    {
        question: "Is the Meta Quest headset included?",
        answer: "Our **Meta Quest app** is compatible with **Meta Quest** 2, **Pro**, **3**, and **3s**.\n\nThe device itself needs to be bought separately.\n\nWe recommend the MQ 3S, which we use on a daily basis."
    },
    {
        question: "I bought a phosphobot starter pack. Do I have phospho pro?",
        answer: "Yes, you have access to phospho pro. Please reach out at contact@phospho.ai with your phospho account email and your Discord username."
    }
];

const FAQSection = () => {
    const [openItems, setOpenItems] = useState<number[]>([]);

    const toggleItem = (index: number) => {
        setOpenItems(prev =>
            prev.includes(index)
                ? prev.filter(i => i !== index)
                : [...prev, index]
        );
    };

    const formatAnswer = (answer: string) => {
        return answer.split('\n\n').map((paragraph, index) => (
            <p key={index} className="mb-4 last:mb-0">
                {paragraph.split(/(\*\*.*?\*\*|\[.*?\]\(.*?\))/).map((part, partIndex) => {
                    // Handle bold text
                    if (part.startsWith('**') && part.endsWith('**')) {
                        return (
                            <strong key={partIndex} className="font-semibold">
                                {part.slice(2, -2)}
                            </strong>
                        );
                    }
                    // Handle links [text](url)
                    const linkMatch = part.match(/^\[(.*?)\]\((.*?)\)$/);
                    if (linkMatch) {
                        const [, linkText, linkUrl] = linkMatch;
                        return (
                            <a
                                key={partIndex}
                                href={linkUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-phospho-green hover:text-phospho-green-dark underline transition-colors"
                            >
                                {linkText}
                            </a>
                        );
                    }
                    return part;
                })}
            </p>
        ));
    };

    return (
        <div className="w-full max-w-4xl mx-auto px-8 py-16">
            <div className="space-y-4">
                {faqData.map((item, index) => {
                    const isOpen = openItems.includes(index);
                    return (
                        <div
                            key={index}
                            className="bg-white rounded-xl shadow-card border border-light-gray overflow-hidden"
                        >
                            <button
                                onClick={() => toggleItem(index)}
                                className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-gray-50 transition-colors"
                            >
                                <span className="font-medium text-dark-gray pr-4">
                                    {item.question}
                                </span>
                                <div className="flex-shrink-0">
                                    {isOpen ? (
                                        <Minus className="h-5 w-5 text-medium-gray" />
                                    ) : (
                                        <Plus className="h-5 w-5 text-medium-gray" />
                                    )}
                                </div>
                            </button>

                            {isOpen && (
                                <div className="px-6 pb-6 border-t border-light-gray">
                                    <div className="pt-4 text-medium-gray">
                                        {formatAnswer(item.answer)}
                                    </div>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default FAQSection;