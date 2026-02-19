struct Char {
    char: u32,
    foreground: vec3u,
    background: vec3u
}

fn gammaToLinear(c:f32) -> f32 {
    return select(c / 12.92, pow((c + 0.055) / 1.055, 2.4), c >= 0.04045);
}

// from https://gist.github.com/earthbound19/e7fe15fdf8ca3ef814750a61bc75b5ce
fn rgbToOklab(color: vec3f) -> vec3f {
    let linear = vec3f(
            gammaToLinear(color.r),
            gammaToLinear(color.g),
            gammaToLinear(color.b)
    );

    var lms = linear * mat3x3f(
        0.4122214708, 0.5363325363, 0.0514459929,
        0.2119034982, 0.6806995451, 0.1073969566,
        0.0883024619, 0.2817188376, 0.6299787005
    );

    lms = sign(lms) * pow(abs(lms), vec3f(1./3.));

    return lms * mat3x3f(
        0.2104542553, 0.7936177850, -0.0040720468,
        1.9779984951, -2.4285922050, 0.4505937099,
        0.0259040371, 0.7827717662, -0.8086757660
    );
}



@group(0) @binding(0)
var input: texture_2d<f32>;
@group(0) @binding(1)
var oklab: texture_storage_2d<rgba16float, read_write>;

@compute @workgroup_size(8, 8)
fn to_Oklab(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(oklab);
    if(any(id.xy >= size.xy)) { return; }

    let color = textureLoad(input, id.xy, 0);
    textureStore(oklab, id.xy, vec4f(rgbToOklab(color.rgb), color.a));
}

@group(1) @binding(1)
var<storage, read_write> output: array<Char>;

const font_size = vec2(10, 18);
const cell_size = font_size + vec2(1);

var<workgroup> Ls: array<f32, cell_size.x * cell_size.y>;
var<workgroup> average_lightness: f32;

@compute @workgroup_size(font_size.x, font_size.y)
fn cal_fg_bg(@builtin(local_invocation_id) local_id: vec3u,
        @builtin(local_invocation_index) local_idx: u32,
        @builtin(global_invocation_id) id: vec3u,
        @builtin(workgroup_id) workgroup_id: vec3u) {
    let size = textureDimensions(oklab);

    var color: vec3f;

    if(all(id.xy < size.xy)) {
        let idx = local_id.x + 1 + (local_id.y + 1) * cell_size.x; // start from (1,1)
        color = textureLoad(oklab, id.xy).xyz;

        Ls[idx] = color.x;

        let caniext = vec4(id.xy > vec2(0) & local_id.xy == vec2(0),
                        id.xy < size.xy - vec2(1) & local_id.xy == font_size - vec2(1));

        if(caniext[0]) { Ls[idx - 1] = textureLoad(oklab, id.xy - vec2(1, 0)).x; }
        if(caniext[2]) { Ls[idx + 1] = textureLoad(oklab, id.xy + vec2(1, 0)).x; }
        if(caniext[1]) { Ls[idx - cell_size.x] = textureLoad(oklab, id.xy - vec2(0, 1)).x; }
        if(caniext[3]) { Ls[idx + cell_size.x] = textureLoad(oklab, id.xy + vec2(0, 1)).x; }

        if(all(caniext.xy)) {
            Ls[0] = textureLoad(oklab, id.xy - vec2(1, 1)).x;
        }
        if(all(caniext.zy)) {
            Ls[cell_size.x - 1] = textureLoad(oklab, vec2(id.x + 1, id.y - 1)).x;
        }
        if(all(caniext.xw)) {
            Ls[(cell_size.y - 1) * cell_size.x] = textureLoad(oklab, vec2(id.x - 1, id.y + 1)).x;
        }
        if(all(caniext.zw)) {
            Ls[cell_size.y * cell_size.x - 1] = textureLoad(oklab, id.xy + vec2(1, 1)).x;
        }
    }

    workgroupBarrier();

    if(local_idx == 0) {
        var sum = 0.;
        for(var i = 0u; i < cell_size.x * cell_size.y; i++) {
            sum += Ls[i];
        }
        average_lightness = sum / (cell_size.x * cell_size.y);
    }

    workgroupBarrier();

    if(local_idx == 0) {
        var bg = vec3f();
        var bgcnt = 0u;
        var fg = vec3f();
        var fgcnt = 0u;
        for(var i = 0u; i < font_size.x; i++) {
            for(var j = 0u; j < font_size.y; j++) {
                let pos = id + vec2(i, j);
                if(all(pos < size)) {
                    let color_ij = textureLoad(oklab, pos).xyz;
                    if(color_ij.x <= average_lightness) {
                        bg += color_ij;
                        bgcnt++;
                    }
                    else {
                        fg += color_ij;
                        fgcnt++;
                    }
                }
            }
        }

        let output_size = size / font_size;
        let output_idx = workgroup_id.x + workgroup_id.y * output_size.x;

        output[output_idx].background = oklabToRgb(bg / bgcnt);
        output[output_idx].foreground = oklabToRgb(fg / fgcnt);
    }
}
