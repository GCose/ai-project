import "@/styles/globals.css";
import type { AppProps } from "next/app";
import { Bricolage_Grotesque, Instrument_Sans } from "next/font/google";

const display = Bricolage_Grotesque({
  subsets: ["latin"],
  variable: "--font-bricolage",
});
const body = Instrument_Sans({
  subsets: ["latin"],
  variable: "--font-instrument",
});

export default function App({ Component, pageProps }: AppProps) {
  return (
    <div className={`${display.variable} ${body.variable}`}>
      <Component {...pageProps} />
    </div>
  );
}
