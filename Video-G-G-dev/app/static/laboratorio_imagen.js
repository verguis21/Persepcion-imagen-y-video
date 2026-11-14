'use strict';
// Laboratorio: no se insertan datos de demostración; solo JSON real de la API.
const $ = id => document.getElementById(id);
let respuestaActual = null, archivoUrl = null, fotoActual = null, filtroActual = 'todos';
const CLASES = {person:'Persona', 'cell phone':'Celular', backpack:'Mochila', handbag:'Bolso',
  suitcase:'Maleta', bottle:'Botella', cup:'Vaso', chair:'Silla'};
const buenNumero = v => typeof v === 'number' && Number.isFinite(v);
function texto(id, contenido) { $(id).textContent = String(contenido); }
function num(v,dec=1) { return buenNumero(Number(v)) && v!==null && v!==undefined && v!=='' ? Number(v).toFixed(dec) : '—'; }
function estadoImagen(id, data) {
  const im=$(id), vacio=$(id==='original'?'vacio-original':'vacio-anotada');
  im.hidden=!data;vacio.hidden=!!data;
  if(data)im.src=data;else im.removeAttribute('src');
}
function celda(nombre, valor) {
  const d=document.createElement('div'), k=document.createElement('span'), v=document.createElement('strong');
  k.textContent=nombre;v.textContent=String(valor);d.append(k,v);return d;
}
function etiqueta(nivel) {return nivel==='revision'?'REVISAR':nivel==='observacion'?'OBSERVACIÓN':'SIN INDICIO';}
function señales(r) {
  if(Array.isArray(r?.senales_visuales))return r.senales_visuales;
  return [{nivel:'informativo',titulo:'No se recibieron señales de esta versión',detalle:'Comprueba que se hayan copiado senalizacion.py y analizador_clip.py.'}];
}
function estadoDe(r) {
  return r?.nivel_revision_visual || (señales(r).some(s=>s.nivel==='revision')?'revision':señales(r).some(s=>s.nivel==='observacion')?'observacion':'informativo');
}
function calidadD(r) {
  const p=r?.preprocesamiento || {}, a=p.antes||r?.calidad_original||{}, d=p.despues||{};
  const zona=$('detalle-calidad');zona.replaceChildren();
  const matriz=document.createElement('div');matriz.className='lista-datos';
  matriz.append(celda('Indicador', 'Antes  →  Después'));
  const metricas=[['Luminosidad','luminosidad'],['Contraste','contraste'],['Nitidez estimada','nitidez_laplaciana'],['Zonas oscuras (%)','porcentaje_zonas_oscuras'],['Zonas sobreexpuestas (%)','porcentaje_zonas_sobreexpuestas']];
  for(const [nombre,key] of metricas)matriz.append(celda(nombre, `${num(a[key])} → ${num(d[key])}`));
  zona.append(matriz);
  const operación=document.createElement('p');operación.className='dato-nota';
  operación.textContent=(p.operaciones||[]).length?
    `Correcciones aplicadas: ${p.operaciones.join(', ')}. Los cambios numéricos no garantizan más precisión.`:
    'No se aplicaron correcciones: la evaluación adaptativa conservó la imagen o se eligió modo original.';
  zona.append(operación);
}
function objetosD(r) {
  const caja=$('lista-objetos');caja.replaceChildren();
  const grupos=[...(r?.personas||[]).map(x=>({...x,clase:'person'})),
    ...(r?.celulares||[]),...(r?.otros_objetos||[])];
  if(!grupos.length){caja.textContent='No se reportaron objetos de las clases habilitadas.';return;}
  const conteos={};for(const d of grupos)conteos[d.clase||'desconocido']=(conteos[d.clase||'desconocido']||0)+1;
  for(const [c,n] of Object.entries(conteos)){
    const div=document.createElement('span');div.className='chip';div.textContent=`${CLASES[c]||c} · ${n}`;caja.append(div);
  }
}
function mostrarDetalle(r,indice) {
  if(!r)return;
  fotoActual=r;
  texto('instante',r.segundo_s!=null ? `t=${num(r.segundo_s,2)} s · ${r.frame_id||'fotograma '+indice}`:'Fotografía individual');
  const nivel=estadoDe(r), estado=$('estado-frame');estado.className=`estado-frame ${nivel}`;
  estado.textContent=etiqueta(nivel)+' · EVIDENCIA VISUAL';
  texto('resumen', `${(r.personas||[]).length} personas detectadas, ${(r.poses_sin_asociar||[]).length} poses sin asociar, ${(r.celulares||[]).length} celulares, ${(r.otros_objetos||[]).length} otros objetos. Marcha inestable: no evaluada.`);
  const avisos=$('senales');avisos.replaceChildren();
  for(const s of señales(r)){
    const bloque=document.createElement('article');bloque.className=`senal ${s.nivel||'informativo'}`;
    const h=document.createElement('strong'),p=document.createElement('p');h.textContent=s.titulo||s.codigo;p.textContent=s.detalle||'';bloque.append(h,p);avisos.append(bloque);
  }
  objetosD(r);calidadD(r);
  const lista=$('advertencias');lista.replaceChildren();
  for(const aviso of ((r.advertencias||[]).length?r.advertencias:['No hay advertencias técnicas adicionales.'])){
    const li=document.createElement('li');li.textContent=aviso;lista.append(li);
  }
  const t=r.tiempos_ms||{}, panel=$('detalle-tiempo');panel.replaceChildren();
  for(const [k,v] of Object.entries(t))panel.append(celda(k,`${num(v)} ms`));
  $('json-dato').textContent=JSON.stringify(r,null,2);
}
function seleccionarMuestra(muestra) {
  estadoImagen('original',muestra.original);estadoImagen('anotada',muestra.anotada);
  const r=respuestaActual?.resultado_clip?.resultados?.[muestra.indice_resultado];
  mostrarDetalle(r,muestra.indice_resultado);
  document.querySelectorAll('.muestra').forEach(b=>b.classList.toggle('activo',Number(b.dataset.indice)===muestra.indice_resultado));
  $('descargar-frame').disabled=!muestra.anotada;
}
function refrescarMuestras(){
  const bloque=$('muestras');bloque.replaceChildren();const fotos=respuestaActual?.vista_previa||[];
  let visibles=0;
  for(const m of fotos){
    const r=respuestaActual?.resultado_clip?.resultados?.[m.indice_resultado];
    const nivel=estadoDe(r);
    if(filtroActual!=='todos' && nivel!==filtroActual)continue;
    visibles++;
    const btn=document.createElement('button');btn.type='button';btn.className='muestra '+nivel;
    btn.dataset.indice=String(m.indice_resultado);
    const im=document.createElement('img');im.src=m.anotada;im.alt=`Fotograma ${m.indice_resultado}`;
    const titulo=document.createElement('span');titulo.textContent=`Fotograma ${m.indice_resultado}`;
    const instante=document.createElement('small');instante.textContent=`t=${num(m.segundo_s,2)} s`;
    const estado=document.createElement('em');estado.textContent=etiqueta(nivel);
    btn.append(im,titulo,instante,estado);btn.addEventListener('click',()=>seleccionarMuestra(m));bloque.append(btn);
  }
  if(!visibles){const p=document.createElement('p');p.className='dato-nota';p.textContent='No hay fotogramas de esta categoría en el clip analizado.';bloque.append(p);}
}
function globalEstado(cantidad, observaciones, calidad) {
  const titulo=$('titulo-estado'),desc=$('explicacion-estado'),circulo=$('estado-circulo');
  circulo.className='estado-circulo '+(cantidad?'revision':observaciones?'observacion':'neutral');
  if(cantidad){titulo.textContent=`ALERTA VISUAL: ${cantidad} fotograma(s) requieren revisión`;desc.textContent='Puede existir una postura inusual, un indicio anterior en seguimiento o un celular detectado cerca de una mano. Revisar el detalle; no confirma caída, distracción ni marcha anómala.';circulo.textContent='REVISAR';}
  else if(observaciones){titulo.textContent=`${observaciones} fotograma(s) con observaciones`;desc.textContent='Se observaron objetos, limitaciones visuales o pérdidas no concluyentes. No se confirmó un incidente.';circulo.textContent='OBSERVAR';}
  else{titulo.textContent='Sin indicios detectados con las reglas actuales';desc.textContent='No encontrar indicios no significa ausencia de riesgo. Para analizar marcha se necesita seguimiento temporal.';circulo.textContent='NO CONCLUYENTE';}
}
function mostrarResultado(res) {
  respuestaActual=res; const fotos=res.vista_previa||[], clip=res.resultado_clip;
  if(clip){
    const r=clip.resultados||[], suma=clip.resumen_visual||{};
    texto('total',clip.fotogramas_analizados??r.length);
    texto('alertas',suma.fotogramas_revision??r.filter(x=>estadoDe(x)==='revision').length);
    texto('rescatados',suma.rescates_roi??r.filter(x=>x.recuperacion_roi?.origen).length);
    texto('tiempo',`${num(clip.tiempo_total_ms,0)} ms`);
    texto('objetos', `${suma.detecciones_celulares_por_fotograma??r.reduce((a,x)=>a+(x.celulares||[]).length,0)} / ${suma.detecciones_otros_objetos_por_fotograma??r.reduce((a,x)=>a+(x.otros_objetos||[]).length,0)}`);
    texto('calidad-contador',suma.fotogramas_calidad_limitada??r.filter(x=>señales(x).some(s=>s.codigo==='calidad')).length);
    texto('meta-clip',`Archivo: ${res.identificacion?.clip_id||'clip'} · Duración ${num(clip.duracion_s,2)} s · ${num(clip.fps_origen,1)} FPS · ${clip.fotogramas_planificados} instantes planificados. Los conteos son detecciones por fotograma, no identidades únicas.`);
    globalEstado(Number($('alertas').textContent),suma.fotogramas_observacion??r.filter(x=>estadoDe(x)==='observacion').length,suma.fotogramas_calidad_limitada||0);
    $('seleccion-panel').hidden=false;filtroActual='todos';
    document.querySelectorAll('#filtros button').forEach(x=>x.classList.toggle('activo',x.dataset.filtro==='todos'));
    refrescarMuestras();
    if(fotos.length)seleccionarMuestra(fotos[0]);
    else{estadoImagen('original',null);estadoImagen('anotada',null);texto('resumen','No se recibieron miniaturas. Revisa la respuesta JSON.');}
    $('visor-video').src=archivoUrl;
  }else{
    $('seleccion-panel').hidden=true; estadoImagen('original',archivoUrl);estadoImagen('anotada',res.imagen_anotada||null);
    const r=res.resultado_imagen||{}; mostrarDetalle(r,0);
    texto('total','1');texto('alertas',estadoDe(r)==='revision'?'1':'0');texto('rescatados','0');
    texto('tiempo',`${num(r.tiempo_procesamiento_ms,0)} ms`);
    texto('objetos',`${(r.celulares||[]).length} / ${(r.otros_objetos||[]).length}`);
    texto('calidad-contador',señales(r).some(s=>s.codigo==='calidad')?'1':'0');
    texto('meta-clip','Fotografía individual: no se puede evaluar cómo camina una persona a partir de una sola imagen.');
    globalEstado(Number($('alertas').textContent),estadoDe(r)==='observacion'?1:0,0);
    $('descargar-frame').disabled=!res.imagen_anotada;
  }
  $('descargar-json').disabled=false;
}
$('formulario').addEventListener('submit',async event=>{
  event.preventDefault();const f=$('archivo').files[0];if(!f)return;
  const clip=f.type.startsWith('video/')||/\.(mp4|avi|mov|mkv)$/i.test(f.name);
  if(f.size>(clip?50:7)*1024*1024){texto('estado','El archivo supera el máximo: 50 MB para clips, 7 MB para fotos.');return;}
  if(archivoUrl) URL.revokeObjectURL(archivoUrl);archivoUrl=URL.createObjectURL(f);
  const fd=new FormData();fd.append(clip?'clip':'frame',f);fd.append('modo',$('modo').value);
  fd.append('metadatos',JSON.stringify(clip?{clip_id:f.name,camara_id:'laboratorio',tiempo_s:0,contexto_video:{}}:{frame_id:f.name,camara_id:'laboratorio',tiempo_s:0,contexto_video:{}}));
  if(clip){fd.append('intervalo_s',$('intervalo').value);fd.append('incluir_muestras','1');}
  else fd.append('incluir_anotada','1');
  const btn=$('analizar');btn.disabled=true;texto('estado','Procesando con los modelos locales…');
  try{
    const response=await fetch(clip?'/api/imagen/v1/clip':'/api/imagen/v1/fotograma',{method:'POST',body:fd});
    const data=await response.json();if(!response.ok||data.estado!=='ok')throw Error(data.mensaje||`HTTP ${response.status}`);
    mostrarResultado(data);texto('estado','Análisis completado. Selecciona un fotograma y revisa sus señales reales.');
  }catch(e){texto('estado',`Error de análisis: ${e.message}`);}finally{btn.disabled=false;}
});
$('filtros').addEventListener('click',e=>{
  if(!e.target.matches('button[data-filtro]'))return;
  filtroActual=e.target.dataset.filtro;
  document.querySelectorAll('#filtros button').forEach(b=>b.classList.toggle('activo',b===e.target));
  refrescarMuestras();
});
function descargar(nombre,contenido,tipo){
  const blob=new Blob([contenido],{type:tipo}),u=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=u;a.download=nombre;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),1000);
}
$('descargar-json').addEventListener('click',()=>{if(respuestaActual)descargar('evidencia_imagen.json',JSON.stringify(respuestaActual,null,2),'application/json');});
$('descargar-frame').addEventListener('click',()=>{
  if(!$('anotada').src)return;const a=document.createElement('a');a.href=$('anotada').src;
  a.download='fotograma_anotado.jpg';document.body.append(a);a.click();a.remove();
});
