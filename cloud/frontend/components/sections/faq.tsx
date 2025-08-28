"use client";

import { useState } from "react";
import { Plus, Minus } from "lucide-react";

interface FAQItem {
  question: string;
  answer: string;
}

const faqData: FAQItem[] = [
  {
    question: "How do I access phospho pro features?",
    answer:
      "Click the **Subscribe** button and enter the same email you used for your Phospho account.\n\nAfter submitting your payment details, you'll be asked to fill in a short form with your **Meta Quest username** and **Discord username**. This helps us sync your access across platforms.\n\nYou'll receive an invite to access the Meta Quest app within 24 hours and you'll be added to a special Discord group.\n\nIf you have any questions, feel free to contact us at contact@phospho.ai.",
  },
  {
    question: "How can I cancel my plan?",
    answer:
      "To manage your subscription, simply use the Stripe link sent to your email when you signed up.\n\nNeed help? Contact us anytime at contact@phospho.ai.",
  },
  {
    question: "Is the Meta Quest headset included?",
    answer:
      "By subscribing, you'll get access to our [Meta Quest app](https://www.meta.com/en-gb/experiences/phospho-teleoperation/8873978782723478/?srsltid=AfmBOorMv4FFiW1uSPvz9cEgsrwhRa5r0-eQ7P-9RRSLcchwzJkBTzoB), which is compatible with **Meta Quest** 2, **Pro**, **3**, and **3s**.\n\nThe device itself needs to be bought separately.\n\nWe recommend the MQ 3S, which we use on a daily basis.",
  },
  {
    question: "I bought a phosphobot starter pack. Do I have phospho pro?",
    answer:
      "Yes, you have access to 1 month of free phospho pro. Please reach out at contact@phospho.ai with your order details and we'll help you set up your account.",
  },
  {
    question:
      "I bought a phosphobot starter pack. Do I need to subscribe to get access to the Meta Quest app?",
    answer:
      "In the phosphobot starter pack, you receive a lifetime access to the Meta Quest app. You don't need to subscribe to use the app.\n\nHowever, you will need to subscribe to access the other phospho pro features, such as training longer and using more advanced models.",
  },
  {
    question: "What robots are compatible?",
    answer:
      "phosphobot is open source and designed to work with virtually any robot.\n\nYou can find the current list of supported robots on our [GitHub](https://github.com/phospho-app/phosphobot).",
  },
  {
    question: "Why does training longer matter?",
    answer:
      'Training time impacts how well your robot learns. Researchers from Stanford, Berkeley, and Meta [recommend training ACT models for "very long" durations.](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?tab=t.0#heading=h.2xiz3mdijyv4) In their experiments, they trained each task for over 5 hours.\n\nSimilarly, NVIDIA\'s Gr00t team trains their models for 10x longer than the phosphobot default.\n\nLonger training require powerful GPUs and expensive hardware, which phosphobot helps you access and manage efficiently.',
  },
];

const FAQSection = () => {
  const [openItems, setOpenItems] = useState<number[]>([]);

  const toggleItem = (index: number) => {
    setOpenItems((prev) =>
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index]
    );
  };

  const formatAnswer = (answer: string) => {
    return answer.split("\n\n").map((paragraph, index) => (
      <p key={index} className="mb-4 last:mb-0">
        {paragraph
          .split(/(\*\*.*?\*\*|\[.*?\]\(.*?\))/)
          .map((part, partIndex) => {
            // Handle bold text
            if (part.startsWith("**") && part.endsWith("**")) {
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
    <div className="w-full max-w-4xl mx-auto px-8 py-4">
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
