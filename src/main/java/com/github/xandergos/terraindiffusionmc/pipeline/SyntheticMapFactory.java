package com.github.xandergos.terraindiffusionmc.pipeline;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.Reader;
import java.util.Arrays;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Generates synthetic climate conditioning maps from Perlin noise quantile-matched
 * to real WorldClim/ETOPO distributions, matching world_pipeline.py make_synthetic_map_factory.
 *
 * <p>Uses precomputed quantile tables from pipeline_data.json (generated once from WorldClim data).
 * Per-world variation comes from the worldSeed used to initialize each channel's FastNoiseLite.
 */
public final class SyntheticMapFactory {

    private static final int N_CHANNELS = 5;
    private static final float[] FREQUENCY_MULT = WorldPipelineModelConfig.frequencyMultipliers();
    private static final float BASE_FREQUENCY = 0.05f;
    private static final int[] OCTAVES = {4, 2, 4, 4, 4};
    private static final float LACUNARITY = 2.0f;
    private static final float GAIN = 0.5f;

    // Loaded from pipeline_data.json (data quantiles are seed-independent WorldClim distributions)
    private final float[][] noiseQuantiles;
    private final float[][] invNoiseQuantilesDx;
    private final float[][] dataQuantiles;
    private final float aTempStd;
    private final float bTempStd;
    private final float tempStdP1;
    private final float tempStdP99;

    // Per-seed noise instances (channels 0..4)
    private final FastNoiseLite[] noises = new FastNoiseLite[N_CHANNELS];

    private static float[][] cachedDataQuantiles;
    private static float cachedATempStd;
    private static float cachedBTempStd;
    private static float cachedTempStdP1;
    private static float cachedTempStdP99;
    private static boolean dataLoaded = false;

    /** @param worldSeed 64-bit world seed (Python: seed & 0xFFFFFFFFFFFFFFFF). Per-channel seeds use lower 32 bits. */
    public SyntheticMapFactory(long worldSeed) {
        loadDataIfNeeded();
        this.dataQuantiles = cachedDataQuantiles;
        this.aTempStd = cachedATempStd;
        this.bTempStd = cachedBTempStd;
        this.tempStdP1 = cachedTempStdP1;
        this.tempStdP99 = cachedTempStdP99;

        this.noiseQuantiles = new float[N_CHANNELS][];
        this.invNoiseQuantilesDx = new float[N_CHANNELS][];
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            FastNoiseLite fnl = new FastNoiseLite((int) ((worldSeed + ch + 1) & 0x7FFFFFFFL));
            fnl.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
            fnl.SetFrequency(BASE_FREQUENCY * FREQUENCY_MULT[ch]);
            fnl.SetFractalType(FastNoiseLite.FractalType.FBm);
            fnl.SetFractalOctaves(OCTAVES[ch]);
            fnl.SetFractalLacunarity(LACUNARITY);
            fnl.SetFractalGain(GAIN);
            noises[ch] = fnl;

            float[] nq = buildNoiseQuantiles(fnl, 64, 1e-4f);
            this.noiseQuantiles[ch] = nq;

            float[] invDx = new float[nq.length - 1];
            for (int i = 0; i < nq.length - 1; i++) {
                invDx[i] = 1.0f / (nq[i + 1] - nq[i]);
            }
            this.invNoiseQuantilesDx[ch] = invDx;
        }
    }

    private static synchronized void loadDataIfNeeded() {
        if (dataLoaded) return;
        try {
            ModelAssetManager.ensureAssetsReady();
            Path pipelineDataPath = ModelAssetManager.resolveAssetPath("pipeline_data.json");
            try (Reader reader = Files.newBufferedReader(pipelineDataPath, StandardCharsets.UTF_8)) {
            JsonObject data = new Gson().fromJson(reader, JsonObject.class);
            int nQ = data.get("n_quantiles").getAsInt();
            JsonArray dataArr = data.getAsJsonArray("data_quantile_tables");
            cachedDataQuantiles = new float[N_CHANNELS][nQ];
            for (int ch = 0; ch < N_CHANNELS; ch++) {
                JsonArray dq = dataArr.get(ch).getAsJsonArray();
                for (int i = 0; i < nQ; i++) {
                    cachedDataQuantiles[ch][i] = dq.get(i).getAsFloat();
                }
            }
            cachedATempStd = data.get("a_temp_std").getAsFloat();
            cachedBTempStd = data.get("b_temp_std").getAsFloat();
            cachedTempStdP1 = data.get("temp_std_p1").getAsFloat();
            cachedTempStdP99 = data.get("temp_std_p99").getAsFloat();
            dataLoaded = true;
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load pipeline_data.json", e);
        }
    }

    /**
     * Compute noise quantile table for a FastNoiseLite instance.
     * Samples on a 1024x1024 grid over [0, 32768) at stride 32, matching Python's
     * _compute_map_stats: x/y in arange(0, 32*1024, 32).
     */
    static float[] buildNoiseQuantiles(FastNoiseLite fnl, int nQuantiles, float eps) {
        float[] values = new float[1024 * 1024];
        int k = 0;
        for (int r = 0; r < 1024; r++)
            for (int c = 0; c < 1024; c++)
                values[k++] = fnl.GetNoise(c * 32, r * 32);
        Arrays.sort(values);

        float[] q = new float[nQuantiles];
        int n = values.length;
        for (int i = 0; i < nQuantiles; i++) {
            float pct = eps + i * (1.0f - 2 * eps) / (nQuantiles - 1);
            float idx = pct * (n - 1);
            int lo = (int) idx;
            int hi = Math.min(lo + 1, n - 1);
            q[i] = values[lo] + (idx - lo) * (values[hi] - values[lo]);
        }

        // Ensure strictly increasing (matches Python build_quantiles)
        float minDiff = Float.MAX_VALUE;
        for (int i = 1; i < nQuantiles; i++)
            if (q[i] > q[i - 1]) minDiff = Math.min(minDiff, q[i] - q[i - 1]);
        if (minDiff == Float.MAX_VALUE) minDiff = 1e-10f;
        for (int i = 1; i < nQuantiles; i++)
            if (q[i] <= q[i - 1]) q[i] = q[i - 1] + minDiff * 0.1f;

        return q;
    }

    /**
     * Sample the synthetic map at world coordinates.
     *
     * @param x1 left world coord (j column in tile space)
     * @param y1 top world coord (i row in tile space)
     * @param x2 exclusive right
     * @param y2 exclusive bottom
     * @return float[5][H][W] where H = y2-y1, W = x2-x1
     *         channels: [elev_sqrt, temp, temp_std, precip, precip_std]
     */
    public float[][][] sample(int x1, int y1, int x2, int y2) {
        int H = y2 - y1;
        int W = x2 - x1;

        // Pre-fetch channel data arrays for inner loops
        FastNoiseLite fnl0 = noises[0], fnl1 = noises[1], fnl2 = noises[2], fnl3 = noises[3], fnl4 = noises[4];
        float[] nq0 = noiseQuantiles[0], dq0 = dataQuantiles[0], inv0 = invNoiseQuantilesDx[0];
        float[] nq1 = noiseQuantiles[1], dq1 = dataQuantiles[1], inv1 = invNoiseQuantilesDx[1];
        float[] nq2 = noiseQuantiles[2], dq2 = dataQuantiles[2], inv2 = invNoiseQuantilesDx[2];
        float[] nq3 = noiseQuantiles[3], dq3 = dataQuantiles[3], inv3 = invNoiseQuantilesDx[3];
        float[] nq4 = noiseQuantiles[4], dq4 = dataQuantiles[4], inv4 = invNoiseQuantilesDx[4];

        int n = noiseQuantiles[0].length;
        float x0_0 = nq0[0], xn1_0 = nq0[n-1], f0_0 = dq0[0], fn1_0 = dq0[n-1];
        float x0_1 = nq1[0], xn1_1 = nq1[n-1], f0_1 = dq1[0], fn1_1 = dq1[n-1];
        float x0_2 = nq2[0], xn1_2 = nq2[n-1], f0_2 = dq2[0], fn1_2 = dq2[n-1];
        float x0_3 = nq3[0], xn1_3 = nq3[n-1], f0_3 = dq3[0], fn1_3 = dq3[n-1];
        float x0_4 = nq4[0], xn1_4 = nq4[n-1], f0_4 = dq4[0], fn1_4 = dq4[n-1];

        float[][][] result = new float[N_CHANNELS][H][W];

        for (int r = 0; r < H; r++) {
            float[] res0 = result[0][r];
            float[] res1 = result[1][r];
            float[] res2 = result[2][r];
            float[] res3 = result[3][r];
            float[] res4 = result[4][r];

            for (int c = 0; c < W; c++) {
                int px = x1 + c;
                int py = y1 + r;

                // Fused quantile transform and noise generation per pixel
                float elev = interpFast(fnl0.GetNoise(px, py), nq0, dq0, inv0, n, x0_0, xn1_0, f0_0, fn1_0);
                float temp = interpFast(fnl1.GetNoise(px, py), nq1, dq1, inv1, n, x0_1, xn1_1, f0_1, fn1_1);
                float tempStd = interpFast(fnl2.GetNoise(px, py), nq2, dq2, inv2, n, x0_2, xn1_2, f0_2, fn1_2);
                float precip = interpFast(fnl3.GetNoise(px, py), nq3, dq3, inv3, n, x0_3, xn1_3, f0_3, fn1_3);
                float precipStd = interpFast(fnl4.GetNoise(px, py), nq4, dq4, inv4, n, x0_4, xn1_4, f0_4, fn1_4);

                // Temp: correct for lapse rate based on elevation
                float lapseRate = -6.5f + 0.0015f * precip;
                lapseRate = lapseRate < -9.8f ? -9.8f : (lapseRate > -4.0f ? -4.0f : lapseRate);
                lapseRate /= 1000.0f;
                temp = temp + lapseRate * (elev > 0.0f ? elev : 0.0f);
                temp = temp < -10.0f ? -10.0f : (temp > 40.0f ? 40.0f : temp);

                // Temp std correction
                float baseline = aTempStd * temp + bTempStd;
                float t01 = (tempStd - tempStdP1) / (tempStdP99 - tempStdP1);
                float baselineClipped = tempStdP1 > -baseline ? tempStdP1 : -baseline;
                tempStd = t01 * (tempStdP99 - baselineClipped) + baselineClipped + baseline;
                tempStd = tempStd > 20.0f ? tempStd : 20.0f;

                // Precip std correction
                float pFactor = (185.0f - 0.04111f * precip) / 185.0f;
                precipStd = precipStd * (pFactor > 0.0f ? pFactor : 0.0f);

                // Elevation: signed sqrt transform
                float elevSqrt = elev < 0 ? -(float)Math.sqrt(-elev) : (float)Math.sqrt(elev);

                res0[c] = elevSqrt;
                res1[c] = temp;
                res2[c] = tempStd;
                res3[c] = precip;
                res4[c] = precipStd;
            }
        }
        return result;
    }

    /** Fast linear interpolation using precomputed inverse dx. */
    static float interpFast(float val, float[] xp, float[] fp, float[] invDx, int n, float x0, float xn1, float f0, float fn1) {
        if (val <= x0) return f0;
        if (val >= xn1) return fn1;
        // Binary search for position
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >>> 1;
            if (xp[mid] <= val) lo = mid; else hi = mid;
        }
        float t = (val - xp[lo]) * invDx[lo];
        return fp[lo] + t * (fp[hi] - fp[lo]);
    }

    /** Linear interpolation matching numpy's np.interp: clamp at boundaries. */
    static float interp(float x, float[] xp, float[] fp) {
        int n = xp.length;
        if (x <= xp[0]) return fp[0];
        if (x >= xp[n - 1]) return fp[n - 1];
        // Binary search for position
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >>> 1;
            if (xp[mid] <= x) lo = mid; else hi = mid;
        }
        float t = (x - xp[lo]) / (xp[hi] - xp[lo]);
        return fp[lo] + t * (fp[hi] - fp[lo]);
    }
}
