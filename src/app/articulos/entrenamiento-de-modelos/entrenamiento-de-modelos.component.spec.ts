import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EntrenamientoDeModelosComponent } from './entrenamiento-de-modelos.component';

describe('EntrenamientoDeModelosComponent', () => {
  let component: EntrenamientoDeModelosComponent;
  let fixture: ComponentFixture<EntrenamientoDeModelosComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ EntrenamientoDeModelosComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(EntrenamientoDeModelosComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
