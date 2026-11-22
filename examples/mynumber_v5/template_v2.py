"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import copy
import os
from pathlib import Path

import cv2
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image

from synthtiger import components, layers, templates, utils

BLEND_MODES = [
    "normal",
    "multiply",
    "screen",
    "overlay",
    "hard_light",
    "soft_light",
    "dodge",
    "divide",
    "addition",
    "difference",
    "darken_only",
    "lighten_only",
]


class DefocusBlur(components.Component):
    def __init__(self, radius=(1, 2)):
        super().__init__()
        if (
            len(radius) != 2
            or not np.all(np.isfinite(radius))
            or not 0 <= radius[0] <= radius[1]
        ):
            raise ValueError("radius must be a finite nonnegative increasing pair")
        self.radius = radius

    def sample(self, meta=None):
        meta = {} if meta is None else meta
        radius = meta["radius"] if "radius" in meta else np.random.uniform(*self.radius)
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        return {"radius": radius}

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        radius = meta["radius"]
        if radius == 0:
            return meta

        extent = int(np.ceil(radius))
        y, x = np.mgrid[-extent : extent + 1, -extent : extent + 1]
        kernel = (x * x + y * y <= radius * radius).astype(np.float32)
        kernel /= kernel.sum()
        for layer in layers:
            layer.image = cv2.filter2D(
                layer.image, -1, kernel, borderType=cv2.BORDER_REFLECT_101
            )
        return meta


class UnsharpMask(components.Component):
    def __init__(self, radius=(0.5, 1.5), percent=(100, 200), threshold=(2, 5)):
        super().__init__()
        ranges = (("radius", radius), ("percent", percent), ("threshold", threshold))
        for name, value in ranges:
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    for item in value
                )
                or not np.all(np.isfinite(value))
                or not 0 <= value[0] <= value[1]
            ):
                raise ValueError(f"{name} must be a finite nonnegative increasing pair")
            setattr(self, name, value)

    def sample(self, meta=None):
        meta = {} if meta is None else meta
        result = {}
        for name in ("radius", "percent", "threshold"):
            value = meta[name] if name in meta else np.random.uniform(*getattr(self, name))
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
            result[name] = value
        return result

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        if meta["radius"] == 0 or meta["percent"] == 0:
            return meta
        for layer in layers:
            image = layer.image.copy()
            rgb = image[..., :3].astype(np.float32)
            blurred = cv2.GaussianBlur(rgb, (0, 0), meta["radius"])
            detail = rgb - blurred
            image[..., :3] = np.clip(
                rgb + np.where(
                    np.abs(detail) > meta["threshold"],
                    detail * meta["percent"] / 100,
                    0,
                ),
                0,
                255,
            )
            layer.image = image
        return meta


class SynthTiger(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        optional = any(
            key in config
            for key in (
                "background_texture",
                "image_color",
                "image_postprocess",
                "character_pad",
                "horizontal_pad",
                "output_size",
                "unsharp",
            )
        )
        self._opt_in = optional
        if optional:
            config = copy.deepcopy(config)
            _resolve_paths(config, Path(__file__).resolve().parents[2])
            _validate_options(config)
        self.background_texture = None
        if "background_texture" in config:
            self.background_texture = components.Switch(
                components.BaseTexture(), **config["background_texture"]
            )
            for paths in self.background_texture.component._paths:
                paths.sort()
        self.image_color = None
        if "image_color" in config:
            self.image_color = components.Gray(**config["image_color"])
        self.character_pad = config.get("character_pad")
        self.horizontal_pad = config.get("horizontal_pad")
        self.output_size = config.get("output_size")
        self.unsharp = None
        if "unsharp" in config:
            self.unsharp = components.Switch(UnsharpMask(), **config["unsharp"])
        self.image_postprocess_scope = config.get("image_postprocess", {}).get(
            "scope", "image"
        )
        self.image_postprocess = None
        if "image_postprocess" in config:
            args = config["image_postprocess"].get("args", [{}, {}, {}])
            self.image_postprocess = components.Iterator(
                [
                    components.Switch(effect, **{"prob": 0, **arg})
                    for effect, arg in zip(
                        [
                            components.MotionBlur(),
                            DefocusBlur(),
                            components.AdditiveGaussianNoise(),
                        ],
                        args,
                    )
                ]
            )

        self.coord_output = config.get("coord_output", True)
        self.mask_output = config.get("mask_output", True)
        self.glyph_coord_output = config.get("glyph_coord_output", True)
        self.glyph_mask_output = config.get("glyph_mask_output", True)
        self.vertical = config.get("vertical", False)
        self.quality = config.get("quality", [95, 95])
        self.visibility_check = config.get("visibility_check", False)
        self.midground = config.get("midground", 0)
        self.midground_offset = components.Translate(
            **config.get("midground_offset", {})
        )
        self.foreground_mask_pad = config.get("foreground_mask_pad", 0)
        self.corpus = components.Selector(
            [
                components.LengthAugmentableCorpus(),
                components.CharAugmentableCorpus(),
            ],
            **config.get("corpus", {}),
        )
        self.font = components.BaseFont(**config.get("font", {}))
        if optional:
            if len(self.font._paths) != len(self.font.paths) or not all(self.font._counts):
                raise ValueError("Every configured font path must exist and contain fonts")
            for group, paths in enumerate(self.font._paths):
                order = sorted(range(len(paths)), key=paths.__getitem__)
                self.font._paths[group] = [paths[index] for index in order]
                self.font._tables[group] = self.font._tables[group][order]
        self.texture = components.Switch(
            components.BaseTexture(), **config.get("texture", {})
        )
        self.colormap2 = components.GrayMap(**config.get("colormap2", {}))
        self.colormap3 = components.GrayMap(**config.get("colormap3", {}))
        self.color = components.Gray(**config.get("color", {}))
        self.shape = components.Switch(
            components.Selector(
                [components.ElasticDistortion(), components.ElasticDistortion()]
            ),
            **config.get("shape", {}),
        )
        self.layout = components.Selector(
            [components.FlowLayout(), components.CurveLayout()],
            **config.get("layout", {}),
        )
        self.style = components.Switch(
            components.Selector(
                [
                    components.TextBorder(),
                    components.TextShadow(),
                    components.TextExtrusion(),
                ]
            ),
            **config.get("style", {}),
        )
        self.transform = components.Switch(
            components.Selector(
                [
                    components.Perspective(),
                    components.Perspective(),
                    components.Trapezoidate(),
                    components.Trapezoidate(),
                    components.Skew(),
                    components.Skew(),
                    components.Rotate(),
                ]
            ),
            **config.get("transform", {}),
        )
        self.fit = components.Fit()
        self.pad = components.Switch(components.Pad(), **config.get("pad", {}))
        self.postprocess = components.Iterator(
            [
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.GaussianBlur()),
                components.Switch(components.Resample()),
                components.Switch(components.MedianBlur()),
            ],
            **config.get("postprocess", {}),
        )

    def generate(self):
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)
        midground = np.random.rand() < self.midground
        background = (
            self.background_texture.sample()
            if self.background_texture is not None else None
        )
        fg_color, fg_style, mg_color, mg_style, bg_color = self._generate_color()
        if background is not None and background["state"] and self.image_color is not None:
            fg_color = self.image_color.sample()

        text = self._generate_text(
            fg_color, fg_style, character_height=self.character_pad is not None
        )
        fg_image, label, bboxes, glyph_fg_image, glyph_bboxes = text[:5]
        if self.character_pad is not None:
            character_height = text[5]
            context = self._context_size(character_height)
            fg_image = np.pad(fg_image, ((context, context), (context, context), (0, 0)))
            glyph_fg_image = np.pad(
                glyph_fg_image, ((context, context), (context, context), (0, 0))
            )
            bboxes = [bbox + [context, context, 0, 0] for bbox in bboxes]
            glyph_bboxes = [bbox + [context, context, 0, 0] for bbox in glyph_bboxes]
        bg_image = self._generate_background(
            fg_image.shape[:2][::-1], bg_color, background
        )

        if midground:
            mg_image, _, _, _, _ = self._generate_text(mg_color, mg_style)
            mg_image = self._erase_image(mg_image, fg_image)
            bg_image = _blend_images(mg_image, bg_image, self.visibility_check)

        if background is not None and background["state"] and self.image_color is not None:
            image = utils.blend_image(fg_image, bg_image, mode="normal")
        else:
            image = _blend_images(fg_image, bg_image, self.visibility_check)
        image, fg_image, glyph_fg_image = self._postprocess_images(
            [image, fg_image, glyph_fg_image],
            shared=background is not None or self.character_pad is not None,
        )
        image_effects = {"metas": [{"state": False, "meta": None} for _ in range(3)]}
        deferred_noise = None
        if (
            self.image_postprocess is not None
            and (
                self.image_postprocess_scope == "all"
                or (background is not None and background["state"])
            )
        ):
            image_effects = self.image_postprocess.sample()
            postprocess_meta = image_effects
            if self.output_size is not None:
                deferred_noise = image_effects["metas"][2]
                postprocess_meta = {
                    "metas": [
                        *image_effects["metas"][:2],
                        {"state": False, "meta": None},
                    ]
                }
            image, fg_image, glyph_fg_image = _image_postprocess(
                [image, fg_image, glyph_fg_image], postprocess_meta
            )

        padding = None
        wide_horizontal_pad = False
        horizontal_range = None
        if self.horizontal_pad is not None:
            wide_horizontal_pad = bool(
                np.random.rand() < self.horizontal_pad.get("prob", 0)
            )
            if wide_horizontal_pad:
                horizontal_range = self.horizontal_pad["range"]
        if self.character_pad is not None:
            (image, fg_image, glyph_fg_image), offset, padding, support = _character_crop(
                [image, fg_image, glyph_fg_image],
                character_height,
                self.character_pad,
                horizontal_range,
            )
            bboxes = [bbox - [*offset, 0, 0] for bbox in bboxes]
            glyph_bboxes = [bbox - [*offset, 0, 0] for bbox in glyph_bboxes]

        output_size = None
        cropped_padding = padding
        if self.output_size is not None:
            (image, fg_image, glyph_fg_image), output_size = _fit_output_size(
                [image, fg_image, glyph_fg_image], self.output_size
            )
            fit_scale = output_size["fit_scale"]
            offset = output_size["offset"]
            bboxes = [bbox * fit_scale + [*offset, 0, 0] for bbox in bboxes]
            glyph_bboxes = [bbox * fit_scale + [*offset, 0, 0] for bbox in glyph_bboxes]
            if "zoom" in output_size:
                bounds = output_size["size"]
                for boxes in (bboxes, glyph_bboxes):
                    for bbox in boxes:
                        start = np.clip(bbox[:2], [0, 0], bounds)
                        end = np.clip(bbox[:2] + bbox[2:], [0, 0], bounds)
                        bbox[:] = [*start, *(end - start)]
            if padding is not None:
                character_height *= fit_scale
                padding, support = _measure_padding(
                    [image, fg_image, glyph_fg_image], character_height
                )
        if deferred_noise is not None and deferred_noise["state"]:
            image = _noise_image([image], deferred_noise["meta"])[0]
        unsharp = None
        if self.unsharp is not None:
            image_layer = layers.Layer(image)
            unsharp = self.unsharp.apply([image_layer])
            image = image_layer.image

        data = {
            "image": image,
            "label": label,
            "quality": quality,
            "mask": fg_image[..., 3],
            "bboxes": bboxes,
            "glyph_mask": glyph_fg_image[..., 3],
            "glyph_bboxes": glyph_bboxes,
        }

        if background is not None:
            data.update(
                background_type="image" if background["state"] else "color",
                background_texture=background,
                image_postprocess=image_effects,
            )
        if output_size is not None:
            data["output_size"] = output_size
            if cropped_padding is not None:
                data["cropped_padding_ratios"] = cropped_padding
        if self.horizontal_pad is not None or output_size is not None:
            data["wide_horizontal_pad"] = wide_horizontal_pad
        if unsharp is not None:
            data["unsharp"] = unsharp
        if self.image_postprocess is not None and background is None:
            data["image_postprocess"] = image_effects
        if padding is not None:
            data.update(
                character_height=character_height,
                padding_ratios=padding,
                foreground_support_bbox=support,
            )

        return data

    def init_save(self, root):
        if self._opt_in and os.path.isdir(root) and os.listdir(root):
            raise ValueError(f"Output directory is not empty: {root}")
        os.makedirs(root, exist_ok=True)

        gt_path = os.path.join(root, "gt.txt")
        coords_path = os.path.join(root, "coords.txt")
        glyph_coords_path = os.path.join(root, "glyph_coords.txt")

        self.gt_file = open(gt_path, "w", encoding="utf-8")
        if self.coord_output:
            self.coords_file = open(coords_path, "w", encoding="utf-8")
        if self.glyph_coord_output:
            self.glyph_coords_file = open(glyph_coords_path, "w", encoding="utf-8")

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        quality = data["quality"]
        mask = data["mask"]
        bboxes = data["bboxes"]
        glyph_mask = data["glyph_mask"]
        glyph_bboxes = data["glyph_bboxes"]

        image = Image.fromarray(image[..., :3].astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        glyph_mask = Image.fromarray(glyph_mask.astype(np.uint8))

        coords = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
        coords = "\t".join([",".join(map(str, map(int, coord))) for coord in coords])
        glyph_coords = [[x, y, x + w, y + h] for x, y, w, h in glyph_bboxes]
        glyph_coords = "\t".join(
            [",".join(map(str, map(int, coord))) for coord in glyph_coords]
        )

        shard = str(idx // 10000)
        image_key = os.path.join("images", shard, f"{idx}.jpg")
        mask_key = os.path.join("masks", shard, f"{idx}.png")
        glyph_mask_key = os.path.join("glyph_masks", shard, f"{idx}.png")
        image_path = os.path.join(root, image_key)
        mask_path = os.path.join(root, mask_key)
        glyph_mask_path = os.path.join(root, glyph_mask_key)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path, quality=quality)
        if self.mask_output:
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            mask.save(mask_path)
        if self.glyph_mask_output:
            os.makedirs(os.path.dirname(glyph_mask_path), exist_ok=True)
            glyph_mask.save(glyph_mask_path)

        self.gt_file.write(f"{image_key}\t{label}\n")
        if self.coord_output:
            self.coords_file.write(f"{image_key}\t{coords}\n")
        if self.glyph_coord_output:
            self.glyph_coords_file.write(f"{image_key}\t{glyph_coords}\n")

    def end_save(self, root):
        self.gt_file.close()
        if self.coord_output:
            self.coords_file.close()
        if self.glyph_coord_output:
            self.glyph_coords_file.close()

    def _generate_color(self):
        mg_color = self.color.sample()
        fg_style = self.style.sample()
        mg_style = self.style.sample()

        if fg_style["state"]:
            fg_color, bg_color, style_color = self.colormap3.sample()
            fg_style["meta"]["meta"]["rgb"] = style_color["rgb"]
        else:
            fg_color, bg_color = self.colormap2.sample()

        return fg_color, fg_style, mg_color, mg_style, bg_color

    def _generate_text(self, color, style, character_height=False):
        label = self.corpus.data(self.corpus.sample())

        # for script using diacritic, ligature and RTL
        chars = utils.split_text(label, reorder=True)

        text = "".join(chars)
        font = self.font.sample({"text": text, "vertical": self.vertical})

        char_layers = [layers.TextLayer(char, **font) for char in chars]
        self.shape.apply(char_layers)
        self.layout.apply(char_layers, {"meta": {"vertical": self.vertical}})
        char_glyph_layers = [char_layer.copy() for char_layer in char_layers]

        text_layer = layers.Group(char_layers).merge()
        text_glyph_layer = text_layer.copy()

        transform = self.transform.sample()
        self.color.apply([text_layer, text_glyph_layer], color)
        self.texture.apply([text_layer, text_glyph_layer])
        self.style.apply([text_layer, *char_layers], style)
        self.transform.apply(
            [text_layer, text_glyph_layer, *char_layers, *char_glyph_layers], transform
        )
        self.fit.apply([text_layer, text_glyph_layer, *char_layers, *char_glyph_layers])
        self.pad.apply([text_layer])

        for char_layer in char_layers:
            char_layer.topleft -= text_layer.topleft
        for char_glyph_layer in char_glyph_layers:
            char_glyph_layer.topleft -= text_layer.topleft

        out = text_layer.output()
        bboxes = [char_layer.bbox for char_layer in char_layers]

        glyph_out = text_glyph_layer.output(bbox=text_layer.bbox)
        glyph_bboxes = [char_glyph_layer.bbox for char_glyph_layer in char_glyph_layers]

        result = out, label, bboxes, glyph_out, glyph_bboxes
        if character_height:
            heights = [
                layer.height
                for char, layer in zip(chars, char_layers)
                if not char.isspace() and np.any(layer.image[..., 3] > 0)
            ]
            if not heights:
                raise RuntimeError("Text has no visible characters")
            result += (float(np.median(heights)),)
        return result

    def _generate_background(self, size, color, background=None):
        layer = layers.RectLayer(size)
        self.color.apply([layer], color)
        self.texture.apply([layer])
        if background is not None and background["state"]:
            texture = self.background_texture.component.data(background["meta"])
            texture = utils.resize_image(texture, size)
            layer.image = utils.blend_image(texture, layer.image, mask=True)
        out = layer.output()
        return out

    def _erase_image(self, image, mask):
        mask = _create_poly_mask(mask, self.foreground_mask_pad)
        mask_layer = layers.Layer(mask)
        image_layer = layers.Layer(image)
        image_layer.bbox = mask_layer.bbox
        self.midground_offset.apply([image_layer])
        out = image_layer.erase(mask_layer).output(bbox=mask_layer.bbox)
        return out

    def _context_size(self, height):
        support = 8
        for switch in self.postprocess.components:
            if switch.prob <= 0:
                continue
            effect = switch.component
            if isinstance(effect, components.GaussianBlur):
                support += int(np.ceil(4 * max(effect.sigma)))
            elif isinstance(effect, components.MedianBlur):
                support += int(max(effect.k))
            elif isinstance(effect, components.Resample):
                support += int(np.ceil(8 / min(effect.size)))
        if self.image_postprocess is not None:
            motion, defocus, _ = self.image_postprocess.components
            if motion.prob > 0:
                support += int(np.ceil(max(motion.component.k) / 2))
            if defocus.prob > 0:
                support += int(np.ceil(max(defocus.component.radius)))
        max_pad = max(1.5, self.character_pad[1])
        if self.horizontal_pad is not None:
            max_pad = max(max_pad, self.horizontal_pad["range"][1])
        return int(np.ceil(max_pad * height)) + support

    def _postprocess_images(self, images, shared=False):
        if shared:
            meta = self.postprocess.sample()
            for effect, switch in zip(self.postprocess.components, meta["metas"]):
                if not switch["state"]:
                    continue
                args = switch["meta"]
                if isinstance(effect.component, components.AdditiveGaussianNoise):
                    images = _noise_image(images, args)
                elif isinstance(effect.component, components.GaussianBlur):
                    images = _shared_augment(images, iaa.GaussianBlur(sigma=args["sigma"]))
                elif isinstance(effect.component, components.Resample):
                    images = _shared_augment(
                        images,
                        iaa.KeepSizeByResize(
                            iaa.Resize(
                                size=args["size"],
                                interpolation=["nearest", "linear", "area", "cubic"],
                            ),
                            interpolation=["nearest", "linear", "area", "cubic"],
                        ),
                    )
                elif isinstance(effect.component, components.MedianBlur):
                    images = _shared_augment(
                        images, iaa.MedianBlur(k=args["k"] * 2 + 1)
                    )
            return images
        image_layers = [layers.Layer(image) for image in images]
        self.postprocess.apply(image_layers)
        outs = [image_layer.output() for image_layer in image_layers]
        return outs


def _shared_augment(images, augmenter):
    augmenter = augmenter.to_deterministic()
    return [
        augmenter(image=image.astype(np.uint8)).astype(np.float32) for image in images
    ]


def _noise_image(images, meta):
    layer = layers.Layer(images[0])
    components.AdditiveGaussianNoise().apply([layer], meta)
    return [layer.image, *images[1:]]


def _image_postprocess(images, meta):
    motion, defocus, noise = meta["metas"]
    if motion["state"]:
        images = _shared_augment(images, iaa.MotionBlur(**motion["meta"]))
    if defocus["state"]:
        image_layers = [layers.Layer(image) for image in images]
        DefocusBlur().apply(image_layers, defocus["meta"])
        images = [layer.image for layer in image_layers]
    if noise["state"]:
        images = _noise_image(images, noise["meta"])
    return images


def _character_crop(images, height, pad_range, horizontal_range=None):
    support = (images[1][..., 3] > 0) | (images[2][..., 3] > 0)
    ys, xs = np.nonzero(support)
    if not len(xs):
        raise RuntimeError("Postprocessing removed all foreground")
    low = int(np.ceil(pad_range[0] * height))
    high = max(low, int(np.floor(pad_range[1] * height)))
    top, right, bottom, left = np.random.randint(low, high + 1, size=4)
    if horizontal_range is not None:
        low = int(np.ceil(horizontal_range[0] * height))
        high = max(low, int(np.floor(horizontal_range[1] * height)))
        right, left = np.random.randint(low, high + 1, size=2)
    x, y = int(xs.min()) - left, int(ys.min()) - top
    width = int(xs.max() - xs.min() + 1)
    ink_height = int(ys.max() - ys.min() + 1)
    outs = [
        image[y : y + ink_height + top + bottom, x : x + width + left + right].copy()
        for image in images
    ]
    ratios = [float(value / height) for value in (top, right, bottom, left)]
    return outs, (x, y), ratios, [int(left), int(top), width, ink_height]


def _measure_padding(images, height):
    support = (images[1][..., 3] > 0) | (images[2][..., 3] > 0)
    ys, xs = np.nonzero(support)
    if not len(xs):
        raise RuntimeError("Resizing removed all foreground")
    x, y = int(xs.min()), int(ys.min())
    width, ink_height = int(xs.max()) - x + 1, int(ys.max()) - y + 1
    canvas_height, canvas_width = support.shape
    margins = (y, canvas_width - x - width, canvas_height - y - ink_height, x)
    return [float(value / height) for value in margins], [x, y, width, ink_height]


def _fit_output_size(images, config):
    scale = float(np.random.uniform(*config["scale"]))
    width, height = [int(round(value * scale)) for value in config["base"]]
    source_height, source_width = images[0].shape[:2]
    fit_scale = min(width / source_width, height / source_height)
    if "zoom" in config:
        zoom = float(np.random.uniform(*config["zoom"]))
        fit_scale *= zoom
    offset = [
        (width - source_width * fit_scale) / 2,
        (height - source_height * fit_scale) / 2,
    ]
    transform = np.array(
        [[fit_scale, 0, offset[0]], [0, fit_scale, offset[1]]], dtype=np.float32
    )
    source = images[0]
    support = (images[1][..., 3] > 0) | (images[2][..., 3] > 0)
    perimeter = np.zeros(support.shape, dtype=bool)
    perimeter[[0, -1], :] = True
    perimeter[:, [0, -1]] = True
    edge_y, edge_x = np.nonzero(perimeter)
    background_edge_source = source.copy()
    background_edge = ~support[edge_y, edge_x]
    if np.any(~background_edge):
        if not np.any(background_edge):
            raise RuntimeError("Output sizing needs background pixels on the crop perimeter")
        background_edge_points = np.column_stack((edge_y[background_edge], edge_x[background_edge]))
        for y, x in zip(edge_y[~background_edge], edge_x[~background_edge]):
            nearest = background_edge_points[np.argmin(np.sum((background_edge_points - (y, x)) ** 2, axis=1))]
            background_edge_source[y, x] = source[tuple(nearest)]
    image = cv2.warpAffine(
        source, transform, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    if np.any(~background_edge):
        background = cv2.warpAffine(
            background_edge_source, transform, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        ys, xs = np.indices((height, width))
        source_x = (xs - offset[0]) / fit_scale
        source_y = (ys - offset[1]) / fit_scale
        outside = (
            (source_x < 0) | (source_x > source_width - 1)
            | (source_y < 0) | (source_y > source_height - 1)
        )
        image[outside] = background[outside]
    outs = [image] + [
        cv2.warpAffine(
            source, transform, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        for source in images[1:]
    ]
    meta = {
        "base": list(config["base"]),
        "scale": scale,
        "size": [width, height],
        "fit_scale": fit_scale,
        "offset": offset,
    }
    if "zoom" in config:
        meta["zoom"] = zoom
    return outs, meta


def _resolve_paths(value, root):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "paths":
                value[key] = [
                    str(root / path) if not os.path.isabs(path) else path
                    for path in item
                ]
            elif key == "augmentation_charset" and item:
                value[key] = str(root / item) if not os.path.isabs(item) else item
            else:
                _resolve_paths(item, root)
    elif isinstance(value, list):
        for item in value:
            _resolve_paths(item, root)


def _validate_options(config):
    def number(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
        )

    def pair(value, name, minimum=0):
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(not number(item) for item in value)
            or not minimum <= value[0] <= value[1]
        ):
            raise ValueError(f"{name} must be a finite increasing pair >= {minimum}")

    def switch(value, name):
        if not isinstance(value, dict) or set(value) - {"prob", "args"}:
            raise ValueError(f"Invalid {name} switch")
        prob = value.get("prob", 0)
        if not number(prob) or not 0 <= prob <= 1:
            raise ValueError(f"{name}.prob must be between zero and one")
        args = value.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"{name}.args must be a mapping")
        return args

    if "output_size" in config:
        value = config["output_size"]
        if (
            not isinstance(value, dict)
            or not {"base", "scale"} <= set(value)
            or set(value) - {"base", "scale", "zoom"}
        ):
            raise ValueError("output_size needs base, scale, and optional zoom")
        base = value["base"]
        if (
            not isinstance(base, (list, tuple))
            or len(base) != 2
            or any(not number(v) or v <= 0 or int(v) != v for v in base)
        ):
            raise ValueError("output_size.base must contain positive integer width and height")
        pair(value["scale"], "output_size.scale")
        if value["scale"][0] <= 0 or any(round(v * value["scale"][0]) < 1 for v in base):
            raise ValueError("output_size.scale must produce positive canvas dimensions")
        if "zoom" in value:
            pair(value["zoom"], "output_size.zoom")
            if value["zoom"][0] <= 0:
                raise ValueError("output_size.zoom must be positive")
    if "horizontal_pad" in config:
        value = config["horizontal_pad"]
        if (
            not isinstance(value, dict)
            or set(value) - {"prob", "range"}
            or "range" not in value
        ):
            raise ValueError("horizontal_pad needs a range and optional prob")
        if "character_pad" not in config:
            raise ValueError("horizontal_pad requires character_pad")
        prob = value.get("prob", 0)
        if not number(prob) or not 0 <= prob <= 1:
            raise ValueError("horizontal_pad.prob must be between zero and one")
        pair(value["range"], "horizontal_pad.range")
    if "unsharp" in config:
        args = switch(config["unsharp"], "unsharp")
        if set(args) - {"radius", "percent", "threshold"}:
            raise ValueError("Invalid unsharp arguments")
        UnsharpMask(**args)
    if "character_pad" in config:
        pair(config["character_pad"], "character_pad")
    if "background_texture" in config:
        value = config["background_texture"]
        args = switch(value, "background_texture")
        if value.get("prob", 1) > 0:
            paths, weights = args.get("paths", []), args.get("weights", [])
            if (
                len(paths) != len(weights)
                or not paths
                or not np.all(np.isfinite(weights))
                or min(weights) < 0
                or sum(weights) <= 0
            ):
                raise ValueError("background_texture needs paths and nonnegative matching weights")
            for path in paths:
                images = (
                    utils.search_files(path, exts=[".jpg", ".jpeg", ".png", ".bmp"])
                    if os.path.isdir(path) else [path] if os.path.isfile(path) else []
                )
                if not images:
                    raise ValueError(f"No background images at {path}")
        if "alpha" in args:
            pair(args["alpha"], "background_texture.alpha")
            if args["alpha"][1] > 1:
                raise ValueError("background_texture.alpha must not exceed one")
        for name in ("grayscale", "crop"):
            if name in args and (not np.isfinite(args[name]) or not 0 <= args[name] <= 1):
                raise ValueError(f"background_texture.{name} must be between zero and one")
    if "image_color" in config:
        args = config["image_color"]
        for name in ("gray", "alpha"):
            if name in args:
                pair(args[name], f"image_color.{name}")
        if "gray" in args and (
            args["gray"][1] > 255 or any(int(value) != value for value in args["gray"])
        ):
            raise ValueError("image_color.gray must be integers between zero and 255")
        if "alpha" in args and args["alpha"][1] > 1:
            raise ValueError("image_color.alpha must not exceed one")
        if "colorize" in args and (
            not np.isfinite(args["colorize"]) or not 0 <= args["colorize"] <= 1
        ):
            raise ValueError("image_color.colorize must be between zero and one")
    if "image_postprocess" in config:
        value = config["image_postprocess"]
        if (
            not isinstance(value, dict)
            or set(value) - {"args", "scope"}
            or value.get("scope", "image") not in ("all", "image")
            or not isinstance(value.get("args", [{}, {}, {}]), (list, tuple))
            or len(value.get("args", [{}, {}, {}])) != 3
        ):
            raise ValueError("image_postprocess.args must contain motion, defocus and noise switches")
        for index, value in enumerate(value.get("args", [{}, {}, {}])):
            args = switch(value, "image_postprocess")
            names = ({"k", "angle"}, {"radius"}, {"scale", "per_channel"})[index]
            if not isinstance(args, dict) or set(args) - names:
                raise ValueError("Invalid image_postprocess effect arguments")
            for name in names - {"per_channel"}:
                if name in args:
                    minimum = 3 if name == "k" else -np.inf if name == "angle" else 0
                    pair(args[name], f"image_postprocess.{name}", minimum=minimum)
            if "k" in args and any(int(k) != k for k in args["k"]):
                raise ValueError("Motion blur k must be integers")
            if "per_channel" in args and (
                not number(args["per_channel"])
                or not 0 <= args["per_channel"] <= 1
            ):
                raise ValueError("Noise per_channel must be between zero and one")


def _blend_images(src, dst, visibility_check=False):
    blend_modes = np.random.permutation(BLEND_MODES)

    for blend_mode in blend_modes:
        out = utils.blend_image(src, dst, mode=blend_mode)
        if not visibility_check or _check_visibility(out, src[..., 3]):
            break
    else:
        raise RuntimeError("Text is not visible")

    return out


def _check_visibility(image, mask):
    gray = utils.to_gray(image[..., :3]).astype(np.uint8)
    mask = mask.astype(np.uint8)
    height, width = mask.shape

    peak = (mask > 127).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bound = (mask > 0).astype(np.uint8)
    bound = cv2.dilate(bound, kernel, iterations=1)

    visit = bound.copy()
    visit ^= 1
    visit = np.pad(visit, 1, constant_values=1)

    border = bound.copy()
    border[mask > 0] = 0

    flag = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    for y in range(height):
        for x in range(width):
            if peak[y][x]:
                cv2.floodFill(gray, visit, (x, y), 1, 16, 16, flag)

    visit = visit[1:-1, 1:-1]
    count = np.sum(visit & border)
    total = np.sum(border)
    return total > 0 and count <= total * 0.1


def _create_poly_mask(image, pad=0):
    height, width = image.shape[:2]
    alpha = image[..., 3].astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.float32)

    cts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cts = sorted(cts, key=lambda ct: sum(cv2.boundingRect(ct)[:2]))

    if len(cts) == 1:
        hull = cv2.convexHull(cts[0])
        cv2.fillConvexPoly(mask, hull, 255)

    for idx in range(len(cts) - 1):
        pts = np.concatenate((cts[idx], cts[idx + 1]), axis=0)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    mask = utils.dilate_image(mask, pad)
    out = utils.create_image((width, height))
    out[..., 3] = mask
    return out
