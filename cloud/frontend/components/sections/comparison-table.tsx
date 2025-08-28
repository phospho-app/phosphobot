import { Check, X } from "lucide-react";

export default function ComparisonTable() {
  const features = [
    { name: "Control robots", free: true, pro: true },
    { name: "AI model training", free: true, pro: true },
    { name: "AI model inference", free: true, pro: true },
    {
      name: (
        <>
          <div>VR Control with Meta Quest 2, Pro, 3, 3s</div>
          <div className="text-xs text-medium-gray">
            Access the{" "}
            <a
              href="https://www.meta.com/en-gb/experiences/phospho-teleoperation/8873978782723478/?srsltid=AfmBOorMv4FFiW1uSPvz9cEgsrwhRa5r0-eQ7P-9RRSLcchwzJkBTzoB"
              target="_blank"
              rel="noopener noreferrer"
              className="text-phospho-green hover:text-phospho-green-dark underline transition-colors"
            >
              phospho teleoperation app
            </a>{" "}
            to control your robot in VR.
          </div>
        </>
      ),
      free: false,
      pro: true,
    },
    { name: "Trainings per month", free: "3", pro: "100" },
    { name: "Max training duration", free: "1h", pro: "3h" },
    { name: "Max number of parallel AI trainings", free: "1", pro: "8" },
    {
      name: (
        <>
          <div>Private mode</div>
          <div className="text-xs text-medium-gray">
            Store datasets as private on Hugging Face by default, train models
            on private datasets, and keep your trained models private.
          </div>
        </>
      ),
      free: false,
      pro: true,
    },
    {
      name: "Private channel on Discord with the team",
      free: false,
      pro: true,
    },
    { name: "phospho pro Discord badge", free: false, pro: true },
  ];

  const renderFeatureValue = (value: boolean | string) => {
    if (typeof value === "boolean") {
      return value ? (
        <Check className="w-5 h-5 text-green-500" />
      ) : (
        <X className="w-5 h-5 text-medium-gray" />
      );
    }
    return <span className="font-semibold text-dark-gray">{value}</span>;
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-[0_10px_15px_rgba(0,0,0,0.1)] overflow-hidden border border-light-gray">
        {/* Header Row */}
        <div className="grid grid-cols-3 items-center bg-white border-b border-light-gray">
          <div className="p-6 invisible">Features</div>
          <div className="p-6 text-center text-dark-gray">Free</div>
          <div className="p-6 text-center font-bold text-dark-gray bg-green-50">
            phospho pro
          </div>
        </div>

        {/* Feature Rows */}
        {features.map((feature, idx) => (
          <div
            key={idx}
            className={
              "grid grid-cols-3 items-center border-b border-light-gray " +
              (idx === features.length - 1 ? "border-b-0" : "")
            }
          >
            <div className="p-4 flex flex-col justify-center min-h-[60px]">
              {feature.name}
            </div>
            <div className="p-4 text-center flex justify-center items-center min-h-[60px]">
              {renderFeatureValue(feature.free)}
            </div>
            <div className="p-4 bg-green-50 text-center flex justify-center items-center min-h-[60px] h-full">
              {renderFeatureValue(feature.pro)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
