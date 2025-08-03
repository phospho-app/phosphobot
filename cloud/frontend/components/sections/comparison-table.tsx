import { Check, X } from 'lucide-react';

export default function ComparisonTable() {
    const features = [
        {
            name: "Control robots",
            free: true,
            pro: true
        },
        {
            name: "AI model training",
            free: true,
            pro: true
        },
        {
            name: "AI model inference",
            free: true,
            pro: true
        },
        {
            name: "VR Control with Meta Quest 2, Pro, 3, 3s",
            free: false,
            pro: true
        },
        {
            name: "Max number of parallel AI trainings",
            free: "1",
            pro: "8"
        },
        {
            name: "Max training duration",
            free: "2h",
            pro: "12h"
        },
        {
            name: "Private channel on Discord with the team",
            free: false,
            pro: true
        },
        {
            name: "phospho pro Discord badge",
            free: false,
            pro: true
        }
    ];

    const renderFeatureValue = (value: boolean | string) => {
        if (typeof value === 'boolean') {
            return value ? (
                <Check className="w-5 h-5 text-primary-green" />
            ) : (
                <X className="w-5 h-5 text-medium-gray" />
            );
        }
        return <span className="font-semibold text-dark-gray">{value}</span>;
    };

    return (
        <div className="w-full max-w-4xl mx-auto">
            <div className="bg-white rounded-xl shadow-card overflow-hidden">
                <div className="grid grid-cols-3 gap-0">
                    {/* Feature names column */}
                    <div className="bg-white border-r border-light-gray">
                        <div className="p-6 border-b border-light-gray">
                            <h3 className="text-xl font-semibold invisible">Features</h3>
                        </div>
                        {features.map((feature, index) => (
                            <div key={index} className="p-4 border-b border-light-gray last:border-b-0 min-h-[80px] flex items-center">
                                <span className="text-sm font-medium text-dark-gray">{feature.name}</span>
                            </div>
                        ))}
                    </div>

                    {/* Free column */}
                    <div className="bg-white border-r border-light-gray">
                        <div className="p-6 border-b border-light-gray text-center">
                            <h3 className="text-xl font-semibold text-dark-gray">Free</h3>
                        </div>
                        {features.map((feature, index) => (
                            <div key={index} className="p-4 border-b border-light-gray last:border-b-0 text-center min-h-[80px] flex items-center justify-center">
                                {renderFeatureValue(feature.free)}
                            </div>
                        ))}
                    </div>

                    {/* Pro column */}
                    <div className="bg-green-100">
                        <div className="p-6 border-b border-green-200 text-center">
                            <h3 className="text-xl font-semibold text-dark-gray">phospho pro</h3>
                        </div>
                        {features.map((feature, index) => (
                            <div key={index} className="p-4 border-b border-green-200 last:border-b-0 text-center min-h-[80px] flex items-center justify-center">
                                {renderFeatureValue(feature.pro)}
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}