###############################################################################
# Spatial Data, GIS, and Remote Sensing
# Vector + Raster Workflows (R)
# Author: Jared Edgerton
# Date: Sys.Date()
#
# Coding lab focus:
#   1) sf workflows: vector data, CRS, reprojection, spatial joins
#   2) terra/raster workflows: reading rasters, cropping, rasterization
#   3) Mapping: plotting rasters + vectors together
#
# Application paper: Harari, M. (2020) AER
# Application paper: Panagopoulos et al. (2023) Land
# Pre-class video: Spatial data formats + coordinate systems
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - Steps are repeated explicitly so students can follow and modify each piece.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed):
#   install.packages(c("sf", "terra", "tidyverse", "tmap"))
#
# Notes:
# - sf is the standard R package for vector spatial data.
# - terra is the modern replacement for the raster package.
# - tmap provides easy thematic mapping.

library(sf)
library(terra)
library(tidyverse)

# Reproducibility
set.seed(123)

# -----------------------------------------------------------------------------
# Part 1: Download + Read a Free Georeferenced Raster (Warm-up)
# -----------------------------------------------------------------------------
# We use the same small test raster from the rasterio repository.

raster_url  <- "https://github.com/rasterio/rasterio/raw/main/tests/data/RGB.byte.tif"
raster_path <- "RGB.byte.tif"

download.file(raster_url, raster_path, mode = "wb")

r <- rast(raster_path)

cat("\n------------------------------\n")
cat("Raster metadata\n")
cat("------------------------------\n")
cat("Path:", raster_path, "\n")
cat("CRS:", crs(r, describe = TRUE)$code, "\n")
print(ext(r))
cat("Dimensions (rows, cols, bands):", nrow(r), ncol(r), nlyr(r), "\n")
cat("Resolution:", res(r), "\n")

# Plot full raster
plotRGB(r, r = 1, g = 2, b = 3, stretch = "lin",
        main = "Remote-sensing raster (RGB.byte.tif) — full image")

# -----------------------------------------------------------------------------
# Part 2: Raster Cropping (Work on a Small Patch)
# -----------------------------------------------------------------------------
# Extract a 256x256 pixel window so processing is fast.

full_ext <- ext(r)
x_res <- res(r)[1]
y_res <- res(r)[2]

# Define window starting at pixel (100, 100), size 256x256
win_col_off <- 100
win_row_off <- 100
win_width   <- 256
win_height  <- 256

xmin_win <- xmin(full_ext) + win_col_off * x_res
xmax_win <- xmin_win + win_width * x_res
ymax_win <- ymax(full_ext) - win_row_off * y_res
ymin_win <- ymax_win - win_height * y_res

crop_ext <- ext(xmin_win, xmax_win, ymin_win, ymax_win)
r_patch <- crop(r, crop_ext)

plotRGB(r_patch, r = 1, g = 2, b = 3, stretch = "lin",
        main = "256x256 raster patch (cropped)")

cat("\nPatch dimensions:", nrow(r_patch), "x", ncol(r_patch), "x", nlyr(r_patch), "\n")

# Store patch bounds for vector creation
patch_ext <- ext(r_patch)
xmin_p <- xmin(patch_ext)
xmax_p <- xmax(patch_ext)
ymin_p <- ymin(patch_ext)
ymax_p <- ymax(patch_ext)

cat("\nPatch bounds:\n")
cat("xmin, xmax:", xmin_p, xmax_p, "\n")
cat("ymin, ymax:", ymin_p, ymax_p, "\n")

# -----------------------------------------------------------------------------
# Part 3: Vector Data in a CRS (sf + creating toy geometries)
# -----------------------------------------------------------------------------
# Create toy "building polygons" (instances) and "roads" (lines) in the raster CRS.

# Step 1: Building polygons
num_buildings <- 40

bx <- runif(num_buildings, min = xmin_p, max = xmax_p)
by <- runif(num_buildings, min = ymin_p, max = ymax_p)
bwidth  <- runif(num_buildings, min = (xmax_p - xmin_p) * 0.01,
                 max = (xmax_p - xmin_p) * 0.05)
bheight <- runif(num_buildings, min = (ymax_p - ymin_p) * 0.01,
                 max = (ymax_p - ymin_p) * 0.05)

building_polys <- vector("list", num_buildings)
for (i in seq_len(num_buildings)) {
  coords <- matrix(c(
    bx[i],            by[i],
    bx[i] + bwidth[i], by[i],
    bx[i] + bwidth[i], by[i] + bheight[i],
    bx[i],            by[i] + bheight[i],
    bx[i],            by[i]
  ), ncol = 2, byrow = TRUE)
  building_polys[[i]] <- st_polygon(list(coords))
}

buildings_sf <- st_sf(
  building_id = seq_len(num_buildings),
  class_name  = rep("building", num_buildings),
  geometry    = st_sfc(building_polys, crs = crs(r))
)

cat("\nBuildings sf (head):\n")
print(head(buildings_sf))

# Step 2: Roads as buffered lines
road_coords <- list(
  # Horizontal road across the middle
  matrix(c(xmin_p, (ymin_p + ymax_p) / 2,
           xmax_p, (ymin_p + ymax_p) / 2), ncol = 2, byrow = TRUE),
  # Vertical road down the middle
  matrix(c((xmin_p + xmax_p) / 2, ymin_p,
           (xmin_p + xmax_p) / 2, ymax_p), ncol = 2, byrow = TRUE),
  # Diagonal road
  matrix(c(xmin_p, ymin_p,
           xmax_p, ymax_p), ncol = 2, byrow = TRUE)
)

road_lines <- lapply(road_coords, st_linestring)
roads_buffer_dist <- (xmax_p - xmin_p) * 0.01
road_polys <- lapply(road_lines, function(ln) st_buffer(st_sfc(ln, crs = crs(r)),
                                                         dist = roads_buffer_dist))

roads_sf <- st_sf(
  road_id    = 1:3,
  class_name = rep("road", 3),
  geometry   = do.call(c, road_polys)
)

cat("\nRoads sf (head):\n")
print(roads_sf)

# Step 3: CRS transformation example (projected CRS -> WGS84)
buildings_wgs84 <- st_transform(buildings_sf, crs = 4326)
roads_wgs84     <- st_transform(roads_sf, crs = 4326)

cat("\nBuildings in WGS84 (head):\n")
print(head(buildings_wgs84))

# Step 4: Spatial join example (points -> buildings)
num_points <- 300
px <- runif(num_points, min = xmin_p, max = xmax_p)
py <- runif(num_points, min = ymin_p, max = ymax_p)

points_sf <- st_sf(
  pt_id    = seq_len(num_points),
  geometry = st_sfc(lapply(seq_len(num_points),
                           function(i) st_point(c(px[i], py[i]))),
                    crs = crs(r))
)

points_joined <- st_join(points_sf, buildings_sf[, "building_id"], join = st_within)
inside_count <- sum(!is.na(points_joined$building_id))
cat("\nPoints inside building polygons:", inside_count, "out of", num_points, "\n")

# -----------------------------------------------------------------------------
# Part 4: Rasterization (Vectors -> Pixel Masks)
# -----------------------------------------------------------------------------
# Semantic mask:
#   0 = background
#   1 = road (stuff)
#   2 = building (thing)

# Create a template raster matching the patch
template <- rast(r_patch[[1]])
values(template) <- 0

# Rasterize roads (value = 1)
roads_vect <- vect(roads_sf)
roads_mask <- rasterize(roads_vect, template, field = 1, background = 0)

# Rasterize buildings (value = 2 for semantic, building_id for instance)
buildings_vect <- vect(buildings_sf)
buildings_sem_mask  <- rasterize(buildings_vect, template, field = 2, background = 0)
buildings_inst_mask <- rasterize(buildings_vect, template, field = "building_id", background = 0)

# Combine semantic mask (roads first, then buildings overwrite)
semantic_mask <- roads_mask
semantic_mask[buildings_sem_mask == 2] <- 2

cat("\nMask summaries:\n")
cat("Semantic classes present:", unique(values(semantic_mask)), "\n")
cat("Instance IDs range:", range(values(buildings_inst_mask), na.rm = TRUE), "\n")

# Plot
par(mfrow = c(1, 3))
plotRGB(r_patch, r = 1, g = 2, b = 3, stretch = "lin", main = "Raster patch")
plot(semantic_mask, main = "Semantic mask\n(0 bg, 1 road, 2 building)")
plot(buildings_inst_mask, main = "Instance mask\n(building IDs)")
par(mfrow = c(1, 1))

# -----------------------------------------------------------------------------
# Part 5: Visualization (Raster + Vector Overlay)
# -----------------------------------------------------------------------------
plotRGB(r_patch, r = 1, g = 2, b = 3, stretch = "lin",
        main = "Raster patch with vector overlay")
plot(st_geometry(roads_sf), col = rgb(1, 0, 0, 0.3), add = TRUE)
plot(st_geometry(buildings_sf), col = rgb(0, 0, 1, 0.3), border = "blue", add = TRUE)
legend("topright", legend = c("Roads", "Buildings"),
       fill = c(rgb(1, 0, 0, 0.3), rgb(0, 0, 1, 0.3)), bty = "n")

# -----------------------------------------------------------------------------
# Part 6: Simple Diagnostics
# -----------------------------------------------------------------------------
# Count pixels per class
sem_vals <- values(semantic_mask)
pixel_counts <- table(sem_vals)
cat("\nPixel counts per semantic class:\n")
print(pixel_counts)

cat("\nFraction of pixels that are background:", mean(sem_vals == 0), "\n")
cat("Fraction of pixels that are road:      ", mean(sem_vals == 1), "\n")
cat("Fraction of pixels that are building:  ", mean(sem_vals == 2), "\n")
