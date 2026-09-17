import { useState } from "react";
import Head from "next/head";
import {
  Loader2,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Sun,
  Moon,
} from "lucide-react";
import Image from "next/image";

interface PredictionResult {
  disease: string;
  confidence: number;
}

const CLASS_NAMES = [
  "Potato___Early_blight",
  "Potato___Late_blight",
  "Potato___healthy",
  "Tomato___Late_blight",
  "Tomato___healthy",
];

// Below this confidence the app says it is not sure instead of guessing
const CONFIDENCE_THRESHOLD = 70;

let modelPromise: Promise<import("@tensorflow/tfjs").GraphModel> | null = null;

const loadModel = async () => {
  const tf = await import("@tensorflow/tfjs");
  if (!modelPromise) modelPromise = tf.loadGraphModel("/model/model.json");
  return { tf, model: await modelPromise };
};

const predictInBrowser = async (src: string): Promise<PredictionResult> => {
  const { tf, model } = await loadModel();
  const img = new window.Image();
  img.src = src;
  await img.decode();

  const probs = tf.tidy(() => {
    const input = tf.image
      .resizeBilinear(tf.browser.fromPixels(img), [224, 224])
      .div(255)
      .expandDims(0);
    return (model.predict(input) as import("@tensorflow/tfjs").Tensor).dataSync();
  });

  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  const confidence = Math.round(probs[best] * 1000) / 10;

  return {
    disease: confidence < CONFIDENCE_THRESHOLD ? "Uncertain" : CLASS_NAMES[best],
    confidence,
  };
};

const diseaseInfo: Record<
  string,
  {
    status: string;
    statusType: "healthy" | "warning" | "danger";
    recommendations: string[][];
  }
> = {
  Uncertain: {
    status: "Not Sure",
    statusType: "warning",
    recommendations: [
      [
        "CropDoc could not identify this with enough confidence",
        "Take a clearer photo of a single leaf in good light",
        "Contact your nearest agricultural extension officer",
      ],
    ],
  },
  Potato___Early_blight: {
    status: "Early Blight Detected",
    statusType: "warning",
    recommendations: [
      [
        "Remove and destroy infected leaves immediately",
        "Apply copper-based fungicide every 7-10 days",
        "Improve air circulation around plants",
        "Avoid overhead watering",
        "Practice crop rotation next season",
      ],
      [
        "Isolate affected plants to prevent spread",
        "Use chlorothalonil-based spray on visible lesions",
        "Reduce leaf wetness by watering in the morning",
        "Clear fallen debris around the base of plants",
        "Increase spacing between plants for better airflow",
      ],
      [
        "Prune lower leaves showing dark spots",
        "Apply neem oil as an organic fungicide alternative",
        "Avoid working with plants when foliage is wet",
        "Check neighboring plants for early signs",
        "Mulch around base to reduce soil splash",
      ],
      [
        "Thin out dense foliage to improve light penetration",
        "Apply a potassium-rich fertilizer to boost immunity",
        "Remove all plant debris at end of season",
        "Avoid using overhead sprinklers",
        "Scout for insect damage that may worsen infection",
      ],
      [
        "Treat with mancozeb at first signs of spotting",
        "Water only at soil level using drip irrigation",
        "Rotate fungicide classes to prevent resistance",
        "Keep a spray log to track treatment intervals",
        "Consult local extension service for regional advice",
      ],
    ],
  },
  Potato___Late_blight: {
    status: "Late Blight Detected",
    statusType: "danger",
    recommendations: [
      [
        "Apply fungicide immediately (mancozeb or chlorothalonil)",
        "Remove all infected plants and tubers",
        "Destroy infected material - do not compost",
        "Monitor surrounding plants closely",
        "Consider harvesting early if disease is severe",
      ],
      [
        "Bag and dispose of all infected tissue off-site",
        "Apply metalaxyl-based systemic fungicide",
        "Avoid moving equipment between infected and healthy areas",
        "Check tubers carefully before storage",
        "Alert neighboring farmers of the outbreak",
      ],
      [
        "Harvest remaining healthy tubers immediately",
        "Drench soil with fungicide to limit spread",
        "Do not replant potatoes in the same area for 3 years",
        "Disinfect all tools used in infected area",
        "Report severe outbreaks to your local agriculture office",
      ],
      [
        "Switch to a systemic fungicide for deeper protection",
        "Avoid irrigation during cool, wet weather",
        "Remove and burn all affected foliage",
        "Monitor tubers in storage for post-harvest rot",
        "Consider resistant varieties for replanting",
      ],
      [
        "Apply phosphonate-based fungicide as a protective measure",
        "Cease irrigation immediately to reduce humidity",
        "Establish a buffer zone around infected plants",
        "Document spread daily to track progression",
        "Consult a plant pathologist if spread is rapid",
      ],
    ],
  },
  Tomato___Late_blight: {
    status: "Late Blight Detected",
    statusType: "danger",
    recommendations: [
      [
        "Apply fungicide immediately",
        "Remove all infected leaves and fruit",
        "Improve air circulation by pruning",
        "Water at base of plants only",
        "Monitor daily for new infections",
      ],
      [
        "Strip all symptomatic foliage and dispose off-site",
        "Apply copper hydroxide spray to remaining foliage",
        "Stake plants higher to reduce ground contact",
        "Avoid touching healthy plants after handling infected ones",
        "Increase inspection frequency to twice daily",
      ],
      [
        "Use a systemic fungicide to protect new growth",
        "Remove affected fruit even if not fully ripe",
        "Sanitize all tools with bleach solution between cuts",
        "Reduce canopy density to allow faster drying",
        "Stop all foliar feeding until infection is controlled",
      ],
      [
        "Harvest all mature fruit immediately as a precaution",
        "Apply mancozeb as a protective barrier spray",
        "Avoid working in the garden after rain",
        "Remove plant ties and supports for cleaning",
        "Plan for resistant tomato varieties next season",
      ],
      [
        "Establish a strict spray schedule every 5-7 days",
        "Remove ground-level leaves to reduce splash infection",
        "Install low-drip irrigation to minimize leaf wetness",
        "Monitor weather forecasts for high-risk blight conditions",
        "Consult an agronomist if more than 30% of crop is affected",
      ],
    ],
  },
  Potato___healthy: {
    status: "Healthy Plant",
    statusType: "healthy",
    recommendations: [
      [
        "Continue current care practices",
        "Monitor regularly for any changes",
        "Ensure adequate watering during dry periods",
        "Apply balanced fertilizer as needed",
        "Practice good garden hygiene",
      ],
      [
        "Keep a regular watering schedule to avoid drought stress",
        "Hill up soil around stems as plants grow",
        "Check leaves weekly for any unusual spots or discoloration",
        "Remove weeds that compete for nutrients",
        "Apply a light mulch to retain soil moisture",
      ],
      [
        "Test soil pH and adjust if outside 5.0–6.5 range",
        "Side-dress with nitrogen fertilizer at flowering",
        "Inspect undersides of leaves for pest eggs",
        "Ensure good drainage to prevent root rot",
        "Avoid over-watering which can attract disease",
      ],
      [
        "Rotate planting location next season to prevent disease buildup",
        "Keep foliage dry by watering early in the day",
        "Scout weekly for Colorado potato beetle",
        "Remove any volunteer potatoes from previous seasons",
        "Apply compost to improve long-term soil health",
      ],
      [
        "Record planting date and variety for future reference",
        "Maintain consistent soil moisture throughout the season",
        "Avoid high-nitrogen fertilizer late in the season",
        "Prepare for harvest when lower leaves begin to yellow",
        "Store harvested tubers in cool, dark conditions",
      ],
    ],
  },
  Tomato___healthy: {
    status: "Healthy Plant",
    statusType: "healthy",
    recommendations: [
      [
        "Continue current care practices",
        "Monitor regularly for any changes",
        "Prune suckers for better air flow",
        "Stake or cage plants for support",
        "Water consistently at the base",
      ],
      [
        "Feed with a phosphorus-rich fertilizer at first flowering",
        "Check for whitefly or aphid colonies under leaves",
        "Tie new growth to supports as plants develop",
        "Mulch around base to keep soil temperature stable",
        "Remove yellowing lower leaves proactively",
      ],
      [
        "Deep-water twice a week rather than shallow daily watering",
        "Pinch out suckers growing in leaf axils",
        "Apply calcium spray to prevent blossom end rot",
        "Ensure at least 6 hours of direct sunlight daily",
        "Keep a garden journal to track growth and issues",
      ],
      [
        "Inspect fruit set and remove any with soft spots early",
        "Rotate crops next season to maintain soil health",
        "Introduce companion plants like basil to deter pests",
        "Apply a balanced liquid fertilizer every two weeks",
        "Check irrigation lines for leaks or blockages",
      ],
      [
        "Thin fruit clusters to 4-5 per truss for larger yields",
        "Watch for early signs of leaf curl due to heat stress",
        "Ensure supports are secure before fruit weight increases",
        "Harvest regularly to encourage continued production",
        "Plan end-of-season soil amendment with compost",
      ],
    ],
  },
};

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [dark, setDark] = useState(false);
  const [recommendationSet, setRecommendationSet] = useState<string[]>([]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;
    setLoading(true);
    try {
      const data = await predictInBrowser(previewUrl);
      setResult(data);
      setRecommendationSet(diseaseInfo[data.disease].recommendations[0]);
    } catch (error) {
      alert("Error: Could not analyze this image. Please try again.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setPreviewUrl("");
    setResult(null);
    setRecommendationSet([]);
  };

  const formatDiseaseName = (disease: string) =>
    disease.replace(/_/g, " ").replace("___", " - ");

  const info = result ? diseaseInfo[result.disease] : null;

  const t = {
    bg: dark ? "bg-[#0d1210]" : "bg-white",
    text: dark ? "text-slate-100" : "text-slate-900",
    muted: dark ? "text-slate-400" : "text-slate-500",
    border: dark ? "border-emerald-700" : "border-emerald-600",
    headerBg: dark ? "bg-[#0d1210]" : "bg-white",
    uploadBg: dark ? "bg-[#111a15]" : "bg-white",
    uploadHover: dark
      ? "hover:bg-emerald-950/40"
      : "group-hover:bg-emerald-50/40",
    uploadIcon: dark ? "bg-[#1a2a20]" : "bg-slate-100",
    card: dark ? "bg-[#111a15]" : "bg-white",
    btnReset: dark
      ? "bg-[#1a2420] border-[#2a3a30] text-slate-300 hover:bg-[#1f2e26]"
      : "bg-white border-slate-200 text-slate-700 hover:bg-slate-50",
    toggleBg: dark
      ? "bg-emerald-900/50 text-emerald-300 hover:bg-emerald-900"
      : "bg-emerald-50 text-emerald-700 hover:bg-emerald-100",
    confBg: dark ? "bg-slate-700" : "bg-slate-200",
  };

  return (
    <>
      <Head>
        <title>CropDoc</title>
        <link rel="shortcut icon" href="/logo.png" type="image/x-icon" />
      </Head>

      <div
        className={`min-h-screen relative overflow-hidden ${t.bg} transition-colors duration-300`}
      >
        <div
          aria-hidden
          className="pointer-events-none absolute -top-30 -right-30 w-180 h-180 rounded-full"
          style={{
            background: dark
              ? "radial-gradient(circle at 60% 40%, rgba(52,211,153,0.18) 0%, rgba(16,185,129,0.08) 40%, transparent 70%)"
              : "radial-gradient(circle at 60% 40%, rgba(52,211,153,0.22) 0%, rgba(16,185,129,0.10) 40%, transparent 70%)",
            filter: "blur(8px)",
          }}
        />

        <div className="relative max-w-7xl mx-auto px-4">
          <header className="py-10">
            <div
              className={`flex items-center justify-between border-b ${t.border} pb-4`}
            >
              <div className="flex items-center gap-3">
                <Image
                  width={40}
                  height={40}
                  src="/logo.png"
                  alt="CropDoc Logo"
                />
                <h1 className={`text-xl font-semibold ${t.text}`}>CropDoc</h1>
              </div>

              <button
                aria-label="Toggle dark mode"
                onClick={() => setDark(!dark)}
                className={`cursor-pointer p-2 rounded-lg transition-colors ${t.toggleBg}`}
              >
                {dark ? (
                  <Sun className="w-4 h-4" />
                ) : (
                  <Moon className="w-4 h-4" />
                )}
              </button>
            </div>
          </header>

          <main className="pb-16">
            <div className="grid lg:grid-cols-2 gap-8">
              <div>
                <div className="sticky top-8">
                  {!previewUrl ? (
                    <label className="block cursor-pointer group">
                      <input
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={handleFileSelect}
                      />
                      <div
                        className={`relative aspect-square border-dashed border ${t.border} rounded-2xl flex flex-col items-center justify-center transition-all ${t.uploadHover} ${t.uploadBg}`}
                      >
                        <div
                          className={`w-16 h-16 ${t.uploadIcon} rounded-2xl flex items-center justify-center mb-6 group-hover:bg-emerald-50 transition-colors`}
                        >
                          <svg
                            className="w-8 h-8 text-slate-400 group-hover:text-emerald-600 transition-colors"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={1.5}
                              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                            />
                          </svg>
                        </div>

                        <h2 className={`text-lg font-medium ${t.text} mb-2`}>
                          Upload crop image
                        </h2>

                        <p className={`text-sm ${t.muted}`}>
                          Click to select or drag and drop
                        </p>
                      </div>
                    </label>
                  ) : (
                    <>
                      <div className="relative aspect-square bg-transparent rounded-2xl overflow-hidden">
                        <Image
                          fill
                          alt="Preview"
                          src={previewUrl}
                          style={{ objectFit: "cover" }}
                        />
                      </div>
                      <button
                        onClick={reset}
                        className={`cursor-pointer mt-4 w-full py-3 px-4 text-sm font-medium border rounded-xl transition-colors ${t.btnReset}`}
                      >
                        Upload different image
                      </button>
                    </>
                  )}
                </div>
              </div>

              <div>
                {!previewUrl ? (
                  <div
                    className={`border-l ${t.border} p-8 h-full flex flex-col items-center justify-center text-center`}
                  >
                    <h3 className={`text-base font-medium ${t.text} mb-2`}>
                      No image selected
                    </h3>
                    <p className={`text-sm ${t.muted} max-w-xs`}>
                      Upload a crop image to analyze it for diseases
                    </p>
                  </div>
                ) : !result ? (
                  <div className={`border-l ${t.border} space-y-4`}>
                    <div className="rounded-2xl p-8">
                      <h3 className={`text-sm font-medium ${t.muted} mb-2`}>
                        Ready to analyze
                      </h3>
                      <p className={`${t.text} mb-6`}>
                        Click the button below to detect diseases in your crop
                        image
                      </p>
                      <button
                        disabled={loading}
                        onClick={analyzeImage}
                        className="cursor-pointer w-full py-4 px-6 text-base font-medium text-white bg-emerald-600 rounded-xl hover:bg-emerald-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                      >
                        {loading ? (
                          <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          "Analyze Image"
                        )}
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className={`border-l ${t.border} pl-4 space-y-12`}>
                    <div>
                      <h3
                        className={`text-base font-semibold ${t.muted} uppercase tracking-wider mb-4`}
                      >
                        Detection Result
                      </h3>
                      <div
                        className={`p-6 ${
                          info?.statusType === "healthy"
                            ? "border-b border-emerald-500"
                            : info?.statusType === "warning"
                              ? "border-b border-amber-500"
                              : "border-b border-rose-500"
                        }`}
                      >
                        <div className="flex items-start gap-4">
                          {info?.statusType === "healthy" ? (
                            <div className="w-12 h-12 bg-emerald-100 rounded-full flex items-center justify-center shrink-0">
                              <CheckCircle2 className="w-6 h-6 text-emerald-600" />
                            </div>
                          ) : info?.statusType === "warning" ? (
                            <div className="w-12 h-12 bg-amber-100 rounded-full flex items-center justify-center shrink-0">
                              <AlertTriangle className="w-6 h-6 text-amber-600" />
                            </div>
                          ) : (
                            <div className="w-12 h-12 bg-rose-100 rounded-full flex items-center justify-center shrink-0">
                              <XCircle className="w-6 h-6 text-rose-600" />
                            </div>
                          )}
                          <div className="flex-1 min-w-0">
                            <p
                              className={`text-lg font-semibold mb-1 ${
                                info?.statusType === "healthy"
                                  ? "text-emerald-600"
                                  : info?.statusType === "warning"
                                    ? "text-amber-500"
                                    : "text-rose-500"
                              }`}
                            >
                              {info?.status}
                            </p>
                            <p className={`text-sm ${t.muted}`}>
                              {formatDiseaseName(result.disease)}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3
                        className={`text-base font-semibold ${t.muted} uppercase tracking-wider mb-4`}
                      >
                        Confidence Score
                      </h3>
                      <div className={`border-b ${t.border} p-6`}>
                        <div className="flex items-center justify-between mb-3">
                          <span className={`text-base ${t.muted}`}>
                            Model confidence
                          </span>
                          <span className={`text-3xl font-bold ${t.text}`}>
                            {result.confidence}%
                          </span>
                        </div>
                        <div className={`w-full ${t.confBg} rounded-full h-2`}>
                          <div
                            className="bg-emerald-600 h-2 rounded-full transition-all duration-1000"
                            style={{ width: `${result.confidence}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3
                        className={`text-base font-semibold ${t.muted} uppercase tracking-wider mb-4`}
                      >
                        Recommendations
                      </h3>
                      <div className={`border-b ${t.border} p-6`}>
                        <ul className="space-y-2">
                          {recommendationSet.map((rec, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-emerald-500 mt-0.5">✓</span>
                              <span
                                className={`text-base ${t.text} leading-relaxed`}
                              >
                                {rec}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </main>
        </div>
      </div>
    </>
  );
}