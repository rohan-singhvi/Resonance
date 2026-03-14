#ifndef MATERIAL_PRESETS_H
#define MATERIAL_PRESETS_H

/*
 * ISO 354 octave-band absorption coefficients for common materials.
 * Bands: 125, 250, 500, 1000, 2000, 4000, 8000 Hz
 *
 * Sources: ISO 354, Mechel "Formulas of Acoustics", Everest "Master Handbook
 * of Acoustics". Values are representative mid-range figures.
 */

#include "simulation.h"
#include <string>
#include <cstring>

static const float BAND_FREQS[NUM_BANDS] = {125, 250, 500, 1000, 2000, 4000, 8000};

namespace MaterialPresets {

struct Entry {
    const char* name;
    float absorption[NUM_BANDS];
};

static const Entry TABLE[] = {
    { "concrete",      { 0.02f, 0.02f, 0.03f, 0.03f, 0.03f, 0.04f, 0.04f } },
    { "brick",         { 0.03f, 0.03f, 0.03f, 0.04f, 0.05f, 0.07f, 0.07f } },
    { "plaster",       { 0.03f, 0.03f, 0.02f, 0.03f, 0.04f, 0.05f, 0.05f } },
    { "wood_panel",    { 0.28f, 0.22f, 0.17f, 0.09f, 0.10f, 0.11f, 0.11f } },
    { "hardwood",      { 0.04f, 0.04f, 0.07f, 0.06f, 0.06f, 0.07f, 0.07f } },
    { "glass",         { 0.35f, 0.25f, 0.18f, 0.12f, 0.07f, 0.04f, 0.04f } },
    { "carpet_thin",   { 0.03f, 0.04f, 0.06f, 0.14f, 0.35f, 0.55f, 0.55f } },
    { "carpet_thick",  { 0.08f, 0.24f, 0.57f, 0.69f, 0.71f, 0.73f, 0.73f } },
    { "acoustic_foam", { 0.10f, 0.25f, 0.55f, 0.90f, 0.95f, 0.95f, 0.95f } },
    { "curtain",       { 0.07f, 0.31f, 0.49f, 0.75f, 0.70f, 0.60f, 0.60f } },
    { "audience",      { 0.60f, 0.74f, 0.88f, 0.96f, 0.93f, 0.85f, 0.85f } },
    { "upholstered",   { 0.19f, 0.37f, 0.56f, 0.67f, 0.61f, 0.59f, 0.59f } },
};

static const int TABLE_SIZE = (int)(sizeof(TABLE) / sizeof(TABLE[0]));

inline bool lookup(const char* name, float out_absorption[NUM_BANDS]) {
    for (int i = 0; i < TABLE_SIZE; ++i) {
        if (strcmp(TABLE[i].name, name) == 0) {
            for (int b = 0; b < NUM_BANDS; ++b)
                out_absorption[b] = TABLE[i].absorption[b];
            return true;
        }
    }
    return false;
}

inline void list_names() {
    for (int i = 0; i < TABLE_SIZE; ++i)
        printf("  %s\n", TABLE[i].name);
}

} // namespace MaterialPresets

#endif
