import { useEffect, useState } from "react";
import Head from "next/head";
import Image from "next/image";
import type { GraphModel, Tensor, Tensor2D } from "@tensorflow/tfjs";
import {
  AlertTriangle,
  ArrowLeft,
  Camera,
  Check,
  CheckCircle2,
  Image as ImageIcon,
  ImageOff,
  Loader2,
  Moon,
  RotateCcw,
  Sun,
  XCircle,
} from "lucide-react";

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

// A disease is only reported at or above this confidence
const CONFIDENCE_THRESHOLD = 70;

// Two checks run before any diagnosis is shown. They never change the
// disease scores; they only decide whether to show them.
//
// 1. Does the photo resemble the leaves the model was trained on?
//    Rejects non-leaf photos (landscapes, screenshots, noise, flat colours).
const LEAF_SIMILARITY_THRESHOLD = 0.6;
// 2. Is it one of the 5 conditions the model knows? A small check trained on
//    33 other PlantVillage classes (maize, pepper, other tomato diseases...).
const KNOWN_THRESHOLD = 0.5;

// Shown when the model finds none of the diseases it knows
const NO_ISSUES = "No_issues";

// Shown when the photo is not something the model was trained to recognise
const UNKNOWN = "Unknown";

// Graph node names for the model outputs: leaf features, disease scores and
// the known-condition score
const FEATURES_NODE = "Identity";
const SCORES_NODE = "Identity_1";
const KNOWN_NODE = "Identity_2";

// The analyzing state stays on screen at least this long, so it never flashes
const MIN_ANALYZE_MS = 1200;

let modelPromise: Promise<{ model: GraphModel; centroids: Tensor2D }> | null =
  null;

// Reports model download progress (0 to 1) to whoever is listening
let reportProgress: (fraction: number) => void = () => {};

const loadModel = async () => {
  const tf = await import("@tensorflow/tfjs");
  if (!modelPromise) {
    modelPromise = Promise.all([
      tf.loadGraphModel("/model/model.json", {
        onProgress: (fraction) => reportProgress(fraction),
      }),
      fetch("/model/centroids.json").then((res) => res.json()),
    ])
      .then(([model, data]) => ({
        model,
        centroids: tf.tensor2d(data.centroids),
      }))
      .catch((err) => {
        modelPromise = null;
        throw err;
      });
  }
  return { tf, ...(await modelPromise) };
};

const predictInBrowser = async (src: string): Promise<PredictionResult> => {
  const { tf, model, centroids } = await loadModel();
  const img = new window.Image();
  img.src = src;
  await img.decode();

  const [simTensor, scoreTensor, knownTensor] = tf.tidy(() => {
    const input = tf.image
      .resizeBilinear(tf.browser.fromPixels(img), [224, 224])
      .div(255)
      .expandDims(0);
    const [features, scores, known] = model.execute(input, [
      FEATURES_NODE,
      SCORES_NODE,
      KNOWN_NODE,
    ]) as Tensor[];
    const unit = features.div(features.norm());
    return [tf.matMul(unit, centroids, false, true).max(), scores, known];
  });
  const similarity = simTensor.dataSync()[0];
  const probs = scoreTensor.dataSync();
  const knownScore = knownTensor.dataSync()[0];
  tf.dispose([simTensor, scoreTensor, knownTensor]);

  if (similarity < LEAF_SIMILARITY_THRESHOLD || knownScore < KNOWN_THRESHOLD) {
    return { disease: UNKNOWN, confidence: 0 };
  }

  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  const confidence = Math.round(probs[best] * 1000) / 10;

  const predicted = CLASS_NAMES[best];
  const isHealthy = predicted.endsWith("___healthy");

  if (!isHealthy && confidence < CONFIDENCE_THRESHOLD) {
    return { disease: NO_ISSUES, confidence };
  }
  return { disease: predicted, confidence };
};

const diseaseInfo: Record<
  string,
  {
    status: string;
    statusType: "healthy" | "warning" | "danger" | "unknown";
    recommendations: string[][];
  }
> = {
  [UNKNOWN]: {
    status: "Can't identify this",
    statusType: "unknown",
    recommendations: [
      [
        "CropDoc knows potato and tomato leaves, and 3 of their diseases",
        "Take a close photo of one leaf in daylight, filling the frame",
        "If the plant looks sick, contact your extension officer",
      ],
    ],
  },
  [NO_ISSUES]: {
    status: "No issues detected",
    statusType: "healthy",
    recommendations: [
      [
        "Keep caring for the plant as you are",
        "Check the leaves again in a few days",
        "If the plant still looks sick, contact your extension officer",
      ],
    ],
  },
  Potato___Early_blight: {
    status: "Early blight detected",
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
    status: "Late blight detected",
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
    status: "Late blight detected",
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
    status: "No issues detected",
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
    status: "No issues detected",
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

const statusColor = {
  unknown: "var(--muted)",
  healthy: "var(--leaf)",
  warning: "var(--rust)",
  danger: "var(--blight)",
};

const splitLabel = (disease: string) => {
  if (disease === NO_ISSUES || disease === UNKNOWN) return null;
  const [crop, condition] = disease.split("___");
  return { crop, condition: condition.replace(/_/g, " ").toLowerCase() };
};

function ConfidenceRing({ value, color }: { value: number; color: string }) {
  const r = 26;
  const c = 2 * Math.PI * r;
  return (
    <div
      className="relative w-16 h-16 shrink-0"
      aria-label={`${value}% confidence`}
    >
      <svg viewBox="0 0 64 64" className="w-16 h-16 -rotate-90">
        <circle
          cx="32"
          cy="32"
          r={r}
          fill="none"
          stroke="var(--line)"
          strokeWidth="6"
        />
        <circle
          cx="32"
          cy="32"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c - (c * value) / 100}
          className="ring-fill"
        />
      </svg>
      <span className="absolute inset-0 flex items-center justify-center font-display text-sm font-semibold">
        {Math.floor(value)}%
      </span>
    </div>
  );
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [dark, setDark] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [recommendationSet, setRecommendationSet] = useState<string[]>([]);

  const [modelReady, setModelReady] = useState(false);
  const [modelProgress, setModelProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Start downloading the model early so the first analysis is fast
  useEffect(() => {
    reportProgress = setModelProgress;
    loadModel()
      .then(() => setModelReady(true))
      .catch(() => {});
  }, []);

  const selectFile = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("That file is not a photo. Choose a photo of a leaf.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    selectFile(e.target.files?.[0]);
    e.target.value = "";
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (!loading) selectFile(e.dataTransfer.files?.[0]);
  };

  const analyzeImage = async () => {
    if (!selectedFile || loading) return;
    setLoading(true);
    setError(null);

    try {
      await loadModel();
      setModelReady(true);
    } catch (err) {
      console.error(err);
      setError(
        "CropDoc could not load. Check your internet connection and try again.",
      );
      setLoading(false);
      return;
    }

    try {
      const [data] = await Promise.all([
        predictInBrowser(previewUrl),
        new Promise((resolve) => setTimeout(resolve, MIN_ANALYZE_MS)),
      ]);
      setResult(data);
      setRecommendationSet(diseaseInfo[data.disease].recommendations[0]);
      window.scrollTo({ top: 0 });
    } catch (err) {
      console.error(err);
      setError("CropDoc could not read this photo. Try another photo.");
    } finally {
      setLoading(false);
    }
  };

  // Back to the "Analyze a leaf" screen
  const reset = () => {
    if (loading) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl("");
    setResult(null);
    setRecommendationSet([]);
    setError(null);
    window.scrollTo({ top: 0 });
  };

  const preparing = loading && !modelReady;

  const info = result ? diseaseInfo[result.disease] : null;
  const label = result ? splitLabel(result.disease) : null;
  const color = info ? statusColor[info.statusType] : "var(--leaf)";
  const isDisease = info
    ? info.statusType === "warning" || info.statusType === "danger"
    : false;

  const fileInput = (useCamera: boolean) => (
    <input
      type="file"
      accept="image/*"
      capture={useCamera ? "environment" : undefined}
      className="sr-only"
      onChange={handleFileSelect}
    />
  );

  const heading = !previewUrl
    ? "Analyze a leaf"
    : preparing
      ? "Getting CropDoc ready"
      : loading
        ? "Analyzing the leaf"
        : "Ready to analyze";

  const intro = !previewUrl
    ? "Use a clear photo of one sick leaf in daylight. CropDoc tells you what is wrong and what to do."
    : preparing
      ? "Loading the AI onto this device. This only happens the first time."
      : loading
        ? "Reading the leaf for signs of disease. This takes a few seconds."
        : "Make sure the leaf fills most of the frame, then analyze it.";

  const loadingLabel = !preparing
    ? "Analyzing"
    : modelProgress > 0
      ? `Getting ready ${Math.round(modelProgress * 100)}%`
      : "Getting ready";

  return (
    <>
      <Head>
        <title>CropDoc</title>
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1, viewport-fit=cover"
        />
        <meta name="theme-color" content={dark ? "#0B2219" : "#EAF2EC"} />
        <link rel="shortcut icon" href="/logo.png" type="image/x-icon" />
      </Head>

      <div data-theme={dark ? "dark" : "light"} className="app-shell">
        <header className="app-header">
          {previewUrl && (
            <button
              onClick={reset}
              disabled={loading}
              aria-label="Start over"
              className="icon-btn lg:hidden"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
          )}
          <div
            className={`items-center gap-2.5 ${previewUrl ? "hidden lg:flex" : "flex"}`}
          >
            <Image
              src="/logo.png"
              alt=""
              width={32}
              height={32}
              className="rounded-lg lg:w-9 lg:h-9"
              priority
            />
            <span className="font-display text-xl lg:text-2xl font-semibold tracking-tight">
              CropDoc
            </span>
          </div>
          <button
            aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
            onClick={() => setDark(!dark)}
            className="icon-btn"
          >
            {dark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
        </header>

        <main className="app-main">
          <section
            className={`visual ${result ? "visual--result" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <div
              className="scanner"
              data-busy={loading}
              data-dragging={dragging}
            >
              <div className="scanner-inner">
                {previewUrl ? (
                  <Image
                    src={previewUrl}
                    alt="Your leaf photo"
                    fill
                    unoptimized
                    className="object-cover"
                  />
                ) : (
                  <label className="w-full h-full flex flex-col items-center justify-center gap-3 cursor-pointer px-8 text-center">
                    {fileInput(false)}
                    <span className="scan-badge">
                      <Camera className="w-7 h-7" />
                    </span>
                    <span className="text-sm text-(--muted) lg:hidden">
                      No photo yet
                    </span>
                    <span className="hidden lg:block text-[15px] text-(--muted)">
                      Drop a leaf photo here
                    </span>
                  </label>
                )}
                {!result && (
                  <>
                    <span className="bracket tl" />
                    <span className="bracket tr" />
                    <span className="bracket bl" />
                    <span className="bracket br" />
                  </>
                )}
                {loading && <span className="scan-line" />}
                {loading && (
                  <span className="scan-status" role="status">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    {loadingLabel}
                  </span>
                )}
              </div>
            </div>
          </section>

          <section className={`panel ${result ? "" : "panel--intro"}`}>
            {!result ? (
              <>
                <h1 className="font-display text-[2.5rem] lg:text-[4rem] leading-[1.02] font-semibold tracking-tight">
                  {heading}
                </h1>
                <p className="mt-2 lg:mt-4 text-[15px] lg:text-lg leading-relaxed text-(--muted) max-w-[40ch]">
                  {intro}
                </p>

                {!previewUrl && (
                  <div className="mt-4 lg:mt-6 flex items-center gap-2 text-sm text-(--muted)">
                    <span>Works with</span>
                    <span className="chip">Potato</span>
                    <span className="chip">Tomato</span>
                  </div>
                )}

                {error && (
                  <p role="alert" className="error-note">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    {error}
                  </p>
                )}

                <div className="hidden lg:flex gap-3 mt-10">
                  {!previewUrl ? (
                    <label className="btn-primary btn-wide">
                      {fileInput(false)}
                      <ImageIcon className="w-5 h-5" />
                      Choose a photo
                    </label>
                  ) : (
                    <>
                      <button
                        disabled={loading}
                        onClick={analyzeImage}
                        className="btn-primary btn-wide"
                      >
                        {loading ? (
                          <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            {loadingLabel}
                          </>
                        ) : (
                          "Analyze leaf"
                        )}
                      </button>
                      <button
                        disabled={loading}
                        onClick={reset}
                        className="btn-secondary btn-wide"
                      >
                        <RotateCcw className="w-5 h-5" />
                        Use another photo
                      </button>
                    </>
                  )}
                </div>
              </>
            ) : (
              <div className="result-sheet">
                <div className="flex items-start gap-4">
                  <span className="status-badge" style={{ background: color }}>
                    {info?.statusType === "unknown" ? (
                      <ImageOff className="w-6 h-6" />
                    ) : info?.statusType === "healthy" ? (
                      <CheckCircle2 className="w-6 h-6" />
                    ) : info?.statusType === "warning" ? (
                      <AlertTriangle className="w-6 h-6" />
                    ) : (
                      <XCircle className="w-6 h-6" />
                    )}
                  </span>
                  <div className="flex-1 min-w-0">
                    <h1 className="font-display text-[1.75rem] lg:text-[2.25rem] leading-tight font-semibold tracking-tight">
                      {info?.status}
                    </h1>
                    {label && (
                      <p className="text-sm lg:text-base text-(--muted) mt-0.5">
                        {isDisease
                          ? `${label.crop}, ${label.condition}`
                          : `${label.crop} leaf`}
                      </p>
                    )}
                  </div>
                  {isDisease && (
                    <ConfidenceRing value={result.confidence} color={color} />
                  )}
                </div>

                <h2 className="font-display text-lg lg:text-xl font-semibold mt-7 lg:mt-8 mb-3 lg:mb-4">
                  {info?.statusType === "unknown" ? "Try this" : "What to do"}
                </h2>
                <ul className="space-y-3">
                  {recommendationSet.map((rec, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <span
                        className="mt-0.5 w-5 h-5 rounded-full flex items-center justify-center shrink-0"
                        style={{
                          background: `color-mix(in srgb, ${color} 16%, transparent)`,
                          color,
                        }}
                      >
                        <Check className="w-3 h-3" strokeWidth={3} />
                      </span>
                      <span className="text-[15px] lg:text-base leading-relaxed">
                        {rec}
                      </span>
                    </li>
                  ))}
                </ul>

                <button
                  onClick={reset}
                  className="hidden lg:flex btn-primary btn-wide w-fit mt-10"
                >
                  Analyze another leaf
                </button>
              </div>
            )}
          </section>
        </main>

        <footer className="action-bar lg:hidden">
          {!previewUrl ? (
            <>
              <label className="btn-primary flex-1">
                {fileInput(true)}
                <Camera className="w-5 h-5" />
                Take a photo
              </label>
              <label className="btn-secondary" aria-label="Choose from gallery">
                {fileInput(false)}
                <ImageIcon className="w-5 h-5" />
              </label>
            </>
          ) : !result ? (
            <>
              <button
                disabled={loading}
                onClick={analyzeImage}
                className="btn-primary flex-1"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    {loadingLabel}
                  </>
                ) : (
                  "Analyze leaf"
                )}
              </button>
              <button
                disabled={loading}
                onClick={reset}
                aria-label="Use another photo"
                className="btn-secondary"
              >
                <RotateCcw className="w-5 h-5" />
              </button>
            </>
          ) : (
            <button onClick={reset} className="btn-primary flex-1">
              Analyze another leaf
            </button>
          )}
        </footer>
      </div>
    </>
  );
}
