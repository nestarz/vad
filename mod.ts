import * as ort from "https://deno.land/x/onnx_runtime@0.0.3/mod.ts";

export type TypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export const resample = (
  array: TypedArray,
  sampleRateOld,
  sampleRateNew
): TypedArray => {
  if (sampleRateNew === sampleRateOld) return array;
  const factor = sampleRateNew / sampleRateOld;
  const newLength = Math.round(array.length * factor);
  const result = new (array.constructor.bind.apply(array.constructor, [
    null,
    newLength,
  ]))();
  for (let i = 0; i < newLength; i++) result[i] = array[Math.floor(i / factor)];
  return result;
};

export interface VADOptions {
  sampleRate: number;
  target16?: boolean;
}

export default async (
  { sampleRate = 8000, target16 = false }: VADOptions = { sampleRate: 8000 }
) => {
  const session = await ort.InferenceSession.create(
    new URL("./models/silero_vad.with_runtime_opt.ort", import.meta.url).href
  );

  const targetSampleRate = target16 ? 16000 : 8000;
  const zeroes = new Float32Array(2 * 64).fill(0);
  const inputs = {
    h: new ort.Tensor("float32", zeroes, [2, 1, 64]),
    c: new ort.Tensor("float32", zeroes, [2, 1, 64]),
    sr: new ort.Tensor("int64", [BigInt(targetSampleRate)]),
  };
  return async (audioFrame: Int16Array) => {
    const data = Float32Array.from(
      resample(new Int16Array(audioFrame.buffer), sampleRate, targetSampleRate),
      (x) => x / 32768.0
    );
    const t = new ort.Tensor("float32", data, [1, data.length]);
    const out = await session.run({ input: t, ...inputs });
    inputs.h = out.hn;
    inputs.c = out.cn;
    const [isSpeech] = out.output.data as Float32Array;
    return isSpeech;
  };
};
